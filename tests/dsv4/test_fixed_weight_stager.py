from __future__ import annotations

import pytest
import torch
from torch import nn

from freetoken.models.deepseek_v4.fixed_weight_stager import FixedWeightStager


class Layer(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[value, value + 1]]), requires_grad=False)

    def forward(self, x):
        return x @ self.weight


def test_stager_keeps_host_weights_and_reuses_one_layer():
    layers = nn.ModuleList([Layer(1), Layer(3)])
    expected = [layer(torch.ones(1, 1)).clone() for layer in layers]
    stager = FixedWeightStager(layers, torch.device("cpu"), budget_bytes=8)

    assert stager.host_bytes == 16
    assert all(weight.device.type == "cpu" for bank in stager._host for weight in bank.values())
    for _ in range(2):
        for layer_id, layer in enumerate(layers):
            stager.stage(layer_id)
            assert layer(torch.ones(1, 1)).equal(expected[layer_id])
            assert stager._active == layer_id
            stager.release(layer_id)
            assert stager._active is None

    stager.close()
    assert stager._host == []


def test_stager_rejects_budget_smaller_than_largest_layer():
    with pytest.raises(ValueError, match="smaller than the largest decoder layer"):
        FixedWeightStager(nn.ModuleList([Layer(1)]), torch.device("cpu"), budget_bytes=7)
