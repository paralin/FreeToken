"""Bounded CPU residency for DeepSeek-V4 decoder-layer fixed weights."""

from __future__ import annotations

import torch
from torch import nn


class FixedWeightStager:
    """Pin layer weights and overlap the next layer's H2D copy with current compute."""

    def __init__(self, layers: nn.ModuleList, device: torch.device, budget_bytes: int):
        self._layers = layers
        self._device = device
        self._copy_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self._host: list[dict[str, nn.Parameter]] = []
        try:
            for layer_id, layer in enumerate(layers):
                weights = {}
                for name, parameter in layer.named_parameters():
                    tensor = parameter.detach()
                    if device.type == "cuda":
                        tensor = tensor.pin_memory()
                        if not tensor.is_pinned():
                            raise RuntimeError(f"failed to pin fixed weight layers.{layer_id}.{name}")
                    weights[name] = nn.Parameter(tensor, requires_grad=False)
                self._assign(layer_id, weights)
                self._host.append(weights)
        except BaseException:
            for layer_id, weights in enumerate(self._host):
                self._assign(layer_id, weights)
            raise

        self.layer_bytes = [
            sum(p.numel() * p.element_size() for p in weights.values())
            for weights in self._host
        ]
        self.required_gpu_bytes = max(
            (left + right for left, right in zip(self.layer_bytes, self.layer_bytes[1:])),
            default=max(self.layer_bytes, default=0),
        )
        if budget_bytes < self.required_gpu_bytes:
            raise ValueError(
                f"fixed-weight GPU budget {budget_bytes} bytes is smaller than the largest "
                f"adjacent decoder-layer pair ({self.required_gpu_bytes} bytes)"
            )
        self._ready: dict[int, tuple[dict[str, nn.Parameter], torch.cuda.Event | None]] = {}
        self._retired: list[tuple[dict[str, nn.Parameter], torch.cuda.Event | None]] = []
        self._active: int | None = None
        self.peak_gpu_layers = 0
        self._closed = False

    @property
    def host_bytes(self) -> int:
        return sum(self.layer_bytes)

    def _assign(self, layer_id: int, weights: dict[str, torch.Tensor]) -> None:
        layer = self._layers[layer_id]
        for name, tensor in weights.items():
            module_name, _, parameter_name = name.rpartition(".")
            module = layer.get_submodule(module_name) if module_name else layer
            module._parameters[parameter_name] = tensor

    def _gpu_set_count(self) -> int:
        return len(self._ready) + len(self._retired)

    def _make_physical_slot(self) -> None:
        if self._gpu_set_count() < 2:
            return
        weights, done = self._retired[0]
        if done is not None:
            # Wait only for the oldest layer's compute event. Its storage can then return to
            # the allocator before the next H2D allocation, enforcing the pair ceiling.
            done.synchronize()
        self._retired.pop(0)
        del weights

    def _prefetch(self, layer_id: int) -> None:
        if layer_id >= len(self._layers) or layer_id in self._ready:
            return
        self._make_physical_slot()
        try:
            if self._copy_stream is None:
                weights = {
                    name: nn.Parameter(tensor.to(self._device), requires_grad=False)
                    for name, tensor in self._host[layer_id].items()
                }
                event = None
            else:
                with torch.cuda.stream(self._copy_stream):
                    weights = {
                        name: nn.Parameter(
                            tensor.to(self._device, non_blocking=True), requires_grad=False
                        )
                        for name, tensor in self._host[layer_id].items()
                    }
                    event = torch.cuda.Event()
                    event.record(self._copy_stream)
            self._ready[layer_id] = (weights, event)
            self.peak_gpu_layers = max(self.peak_gpu_layers, self._gpu_set_count())
        except BaseException:
            self._assign(layer_id, self._host[layer_id])
            raise

    def stage(self, layer_id: int) -> None:
        if self._closed:
            raise RuntimeError("fixed-weight stager is closed")
        if self._active is not None:
            raise RuntimeError(f"fixed-weight layer {self._active} is still active")
        self._prefetch(layer_id)
        weights, ready = self._ready[layer_id]
        compute_stream = None
        if ready is not None:
            compute_stream = torch.cuda.current_stream(self._device)
            compute_stream.wait_event(ready)
        try:
            self._assign(layer_id, weights)
            self._active = layer_id
            self._prefetch(layer_id + 1)
        except BaseException:
            self._assign(layer_id, self._host[layer_id])
            self._active = None
            raise

    def release(self, layer_id: int) -> None:
        if self._active != layer_id:
            raise RuntimeError(f"fixed-weight layer {layer_id} is not active")
        weights, _ = self._ready.pop(layer_id)
        self._assign(layer_id, self._host[layer_id])
        done = None
        if self._device.type == "cuda":
            done = torch.cuda.Event()
            done.record(torch.cuda.current_stream(self._device))
        self._retired.append((weights, done))
        self.peak_gpu_layers = max(self.peak_gpu_layers, self._gpu_set_count())
        self._active = None

    def close(self) -> None:
        if self._closed:
            return
        if self._active is not None:
            self.release(self._active)
        if self._copy_stream is not None:
            self._copy_stream.synchronize()
        for _, done in self._retired:
            if done is not None:
                done.synchronize()
        for layer_id in range(len(self._layers)):
            self._assign(layer_id, self._host[layer_id])
        self._ready.clear()
        self._retired.clear()
        self._host.clear()
        self._layers = nn.ModuleList()
        self._closed = True
