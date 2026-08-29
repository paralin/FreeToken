from __future__ import annotations

import pytest
import torch
from torch import nn

from freetoken.models.deepseek_v4.fixed_weight_stager import FixedWeightStager


class Layer(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[value, value + 1]], dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        return x @ self.weight


def test_stager_keeps_host_weights_and_reuses_one_layer():
    layers = nn.ModuleList([Layer(1), Layer(3)])
    expected = [layer(torch.ones(1, 1)).clone() for layer in layers]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stager = FixedWeightStager(layers, device, budget_bytes=16)
    host_parameters = [layer.weight for layer in layers]

    assert stager.host_bytes == 16
    assert stager.required_gpu_bytes == 16
    if device.type == "cuda":
        assert all(parameter.is_pinned() for parameter in host_parameters)
    assert all(weight.device.type == "cpu" for bank in stager._host for weight in bank.values())
    for _ in range(2):
        for layer_id, layer in enumerate(layers):
            sibling_id = 1 - layer_id
            stager.stage(layer_id)
            assert layer.weight is not host_parameters[layer_id]
            assert layer.weight.device.type == device.type
            assert layers[sibling_id].weight is host_parameters[sibling_id]
            assert layers[sibling_id].weight.device.type == "cpu"
            output = layer(torch.ones(1, 1, device=device)).cpu()
            assert output.equal(expected[layer_id])
            assert stager._active == layer_id
            if layer_id == 0:
                assert set(stager._ready) == {0, 1}
            stager.release(layer_id)
            assert layer.weight is host_parameters[layer_id]
            assert layer_id not in stager._ready
            assert stager._active is None

    assert stager.peak_gpu_layers == 2
    stager.close()
    assert stager._host == []
    assert all(layer.weight is parameter for layer, parameter in zip(layers, host_parameters))


def test_stager_rejects_budget_smaller_than_largest_layer():
    with pytest.raises(ValueError, match="smaller than the largest adjacent decoder-layer pair"):
        FixedWeightStager(nn.ModuleList([Layer(1), Layer(2)]), torch.device("cpu"), budget_bytes=15)


def test_startup_budget_reserves_actual_staged_layer_bytes():
    from freetoken.engine.engine import _startup_kv_budget

    default = _startup_kv_budget(0.9, 10_000, 8_000)
    staged = _startup_kv_budget(0.9, 10_000, 8_000, staging_bytes=1_400)
    assert default == 7_000
    assert staged == default - 1_400


def test_release_after_layer_failure_restores_host_parameters():
    layers = nn.ModuleList([Layer(1), Layer(3)])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stager = FixedWeightStager(layers, device, budget_bytes=16)
    host = [layer.weight for layer in layers]

    stager.stage(0)
    try:
        raise RuntimeError("layer failed")
    except RuntimeError:
        stager.release(0)

    assert layers[0].weight is host[0]
    assert layers[1].weight is host[1]
    assert stager._active is None
    stager.close()
