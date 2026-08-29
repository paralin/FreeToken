"""Bounded CPU residency for DeepSeek-V4 decoder-layer fixed weights."""

from __future__ import annotations

import torch
from torch import nn


class FixedWeightStager:
    """Pin layer weights and overlap the next layer's H2D copy with current compute."""

    def __init__(
        self,
        layers: nn.ModuleList,
        device: torch.device,
        budget_bytes: int,
        resident_bytes: int = 0,
    ):
        self._layers = layers
        self._device = device
        self._copy_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        source = [dict(layer.named_parameters()) for layer in layers]
        self.layer_bytes = [
            sum(p.numel() * p.element_size() for p in weights.values())
            for weights in source
        ]
        resident_ids = []
        resident_total = 0
        for layer_id, layer_bytes in enumerate(self.layer_bytes):
            if resident_total + layer_bytes > resident_bytes:
                break
            resident_ids.append(layer_id)
            resident_total += layer_bytes
        if resident_bytes > 0 and not resident_ids:
            raise ValueError(
                f"fixed-weight resident budget {resident_bytes} bytes cannot fit decoder "
                f"layer 0 ({self.layer_bytes[0]} bytes)"
            )
        self.resident_layer_ids = tuple(resident_ids)
        self.resident_bytes = resident_total
        self._resident = set(resident_ids)

        streamed_ids = [i for i in range(len(layers)) if i not in self._resident]
        streamed_bytes = [self.layer_bytes[i] for i in streamed_ids]
        self.required_gpu_bytes = max(
            (left + right for left, right in zip(streamed_bytes, streamed_bytes[1:])),
            default=max(streamed_bytes, default=0),
        )
        if budget_bytes < self.required_gpu_bytes:
            raise ValueError(
                f"fixed-weight GPU budget {budget_bytes} bytes is smaller than the largest "
                f"adjacent streamed decoder-layer pair ({self.required_gpu_bytes} bytes)"
            )

        self._host: list[dict[str, nn.Parameter] | None] = []
        try:
            for layer_id, weights in enumerate(source):
                if layer_id in self._resident:
                    placed = {
                        name: nn.Parameter(parameter.detach().to(device), requires_grad=False)
                        for name, parameter in weights.items()
                    }
                    self._assign(layer_id, placed)
                    self._host.append(None)
                    continue
                pinned = {}
                for name, parameter in weights.items():
                    tensor = parameter.detach()
                    if device.type == "cuda":
                        tensor = tensor.pin_memory()
                        if not tensor.is_pinned():
                            raise RuntimeError(f"failed to pin fixed weight layers.{layer_id}.{name}")
                    pinned[name] = nn.Parameter(tensor, requires_grad=False)
                self._assign(layer_id, pinned)
                self._host.append(pinned)
        except BaseException:
            for layer_id, weights in enumerate(self._host):
                if weights is not None:
                    self._assign(layer_id, weights)
            raise

        self._ready: dict[int, tuple[dict[str, nn.Parameter], torch.cuda.Event | None]] = {}
        self._retired: list[tuple[dict[str, nn.Parameter], torch.cuda.Event | None]] = []
        self._active: int | None = None
        self._resident_done: dict[int, torch.cuda.Event] = {}
        self.peak_gpu_layers = 0
        self._closed = False

    @property
    def host_bytes(self) -> int:
        return sum(
            self.layer_bytes[layer_id]
            for layer_id, weights in enumerate(self._host)
            if weights is not None
        )

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

    def _next_streamed(self, layer_id: int) -> int | None:
        for candidate in range(layer_id + 1, len(self._layers)):
            if candidate not in self._resident:
                return candidate
        return None

    def _prefetch(self, layer_id: int) -> None:
        if layer_id >= len(self._layers) or layer_id in self._resident or layer_id in self._ready:
            return
        self._make_physical_slot()
        host = self._host[layer_id]
        assert host is not None
        try:
            if self._copy_stream is None:
                weights = {
                    name: nn.Parameter(tensor.to(self._device), requires_grad=False)
                    for name, tensor in host.items()
                }
                event = None
            else:
                with torch.cuda.stream(self._copy_stream):
                    weights = {
                        name: nn.Parameter(
                            tensor.to(self._device, non_blocking=True), requires_grad=False
                        )
                        for name, tensor in host.items()  # type: ignore[union-attr]
                    }
                    event = torch.cuda.Event()
                    event.record(self._copy_stream)
            self._ready[layer_id] = (weights, event)
            self.peak_gpu_layers = max(self.peak_gpu_layers, self._gpu_set_count())
        except BaseException:
            host = self._host[layer_id]
            if host is not None:
                self._assign(layer_id, host)
            raise

    def stage(self, layer_id: int) -> None:
        if self._closed:
            raise RuntimeError("fixed-weight stager is closed")
        if self._active is not None:
            raise RuntimeError(f"fixed-weight layer {self._active} is still active")
        if layer_id in self._resident:
            self._active = layer_id
            next_layer = self._next_streamed(layer_id)
            if next_layer is not None:
                self._prefetch(next_layer)
            return
        self._prefetch(layer_id)
        weights, ready = self._ready[layer_id]
        if ready is not None:
            torch.cuda.current_stream(self._device).wait_event(ready)
        try:
            self._assign(layer_id, weights)
            self._active = layer_id
            next_layer = self._next_streamed(layer_id)
            if next_layer is not None:
                self._prefetch(next_layer)
        except BaseException:
            host = self._host[layer_id]
            if host is not None:
                self._assign(layer_id, host)
            self._active = None
            raise

    def release(self, layer_id: int) -> None:
        if self._active != layer_id:
            raise RuntimeError(f"fixed-weight layer {layer_id} is not active")
        if layer_id in self._resident:
            if self._device.type == "cuda":
                done = torch.cuda.Event()
                done.record(torch.cuda.current_stream(self._device))
                self._resident_done[layer_id] = done
            self._active = None
            return
        weights, _ = self._ready.pop(layer_id)
        host = self._host[layer_id]
        assert host is not None
        self._assign(layer_id, host)
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
        for done in self._resident_done.values():
            done.synchronize()
        for layer_id in self.resident_layer_ids:
            cpu_weights = {
                name: nn.Parameter(parameter.detach().cpu(), requires_grad=False)
                for name, parameter in self._layers[layer_id].named_parameters()
            }
            self._assign(layer_id, cpu_weights)
        for layer_id, weights in enumerate(self._host):
            if weights is not None:
                self._assign(layer_id, weights)
        self._ready.clear()
        self._retired.clear()
        self._resident_done.clear()
        self._host.clear()
        self._layers = nn.ModuleList()
        self._closed = True
