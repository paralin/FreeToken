"""Bounded CPU residency for DeepSeek-V4 decoder-layer fixed weights."""

from __future__ import annotations

import torch
from torch import nn


class FixedWeightStager:
    """Keep layer parameters on CPU and materialize one complete layer for execution."""

    def __init__(self, layers: nn.ModuleList, device: torch.device, budget_bytes: int):
        self._layers = layers
        self._device = device
        self._host = [dict(layer.named_parameters()) for layer in layers]
        self.layer_bytes = [
            sum(p.numel() * p.element_size() for p in weights.values())
            for weights in self._host
        ]
        self.required_gpu_bytes = max(self.layer_bytes, default=0)
        if budget_bytes < self.required_gpu_bytes:
            raise ValueError(
                f"fixed-weight GPU budget {budget_bytes} bytes is smaller than the largest "
                f"decoder layer ({self.required_gpu_bytes} bytes)"
            )
        self._active: int | None = None

    @property
    def host_bytes(self) -> int:
        return sum(self.layer_bytes)

    def _assign(self, layer_id: int, weights: dict[str, torch.Tensor]) -> None:
        layer = self._layers[layer_id]
        for name, tensor in weights.items():
            module_name, _, parameter_name = name.rpartition(".")
            module = layer.get_submodule(module_name) if module_name else layer
            module._parameters[parameter_name] = tensor

    def stage(self, layer_id: int) -> None:
        if self._active is not None:
            raise RuntimeError(f"fixed-weight layer {self._active} is still active")
        try:
            weights = {
                name: nn.Parameter(tensor.to(self._device), requires_grad=False)
                for name, tensor in self._host[layer_id].items()
            }
            self._assign(layer_id, weights)
        except BaseException:
            self._assign(layer_id, self._host[layer_id])
            raise
        self._active = layer_id

    def release(self, layer_id: int) -> None:
        if self._active != layer_id:
            raise RuntimeError(f"fixed-weight layer {layer_id} is not active")
        # Parameter storage cannot return to the allocator until kernels using it finish.
        # This synchronous first implementation keeps exactly one layer resident.
        if self._device.type == "cuda":
            torch.cuda.current_stream(self._device).synchronize()
        self._assign(layer_id, self._host[layer_id])
        self._active = None

    def close(self) -> None:
        if self._active is not None:
            self.release(self._active)
        self._host.clear()
        self._layers = nn.ModuleList()
