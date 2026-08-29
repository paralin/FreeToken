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
        required = max(self.layer_bytes, default=0)
        if budget_bytes < required:
            raise ValueError(
                f"fixed-weight GPU budget {budget_bytes} bytes is smaller than the largest "
                f"decoder layer ({required} bytes)"
            )
        self._active: int | None = None

    @property
    def host_bytes(self) -> int:
        return sum(self.layer_bytes)

    def stage(self, layer_id: int) -> None:
        if self._active is not None:
            raise RuntimeError(f"fixed-weight layer {self._active} is still active")
        weights = {name: tensor.to(self._device) for name, tensor in self._host[layer_id].items()}
        self._layers[layer_id].load_state_dict(weights, assign=True, strict=False)
        self._active = layer_id

    def release(self, layer_id: int) -> None:
        if self._active != layer_id:
            raise RuntimeError(f"fixed-weight layer {layer_id} is not active")
        # Parameter storage cannot return to the allocator until kernels using it finish.
        # This synchronous first implementation keeps exactly one layer resident.
        if self._device.type == "cuda":
            torch.cuda.current_stream(self._device).synchronize()
        self._layers[layer_id].load_state_dict(self._host[layer_id], assign=True, strict=False)
        self._active = None

    def close(self) -> None:
        if self._active is not None:
            if self._device.type == "cuda":
                torch.cuda.current_stream(self._device).synchronize()
            self._active = None
        self._host.clear()
        self._layers = nn.ModuleList()
