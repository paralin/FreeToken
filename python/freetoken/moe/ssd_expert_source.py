"""Bounded row-addressable expert storage backed by safetensor shards."""

from __future__ import annotations

import collections
import json
import os
import struct
import threading
from dataclasses import dataclass

import torch


_DTYPE = {"U8": torch.uint8, "F8_E8M0": torch.float8_e8m0fnu}


@dataclass(frozen=True)
class _TensorRange:
    path: str
    offset: int
    length: int
    shape: tuple[int, ...]
    dtype: torch.dtype


class Dsfp4SafetensorSource:
    """Serve exact DeepSeek-FP4 expert rows with bounded host residency.

    Safetensor headers are indexed at construction. Tensor payloads are read with
    ``pread`` on cache misses, so shards are never mapped into the process. The
    source serializes cache mutation while allowing callers from multiple engine
    threads. ``close`` releases every shard descriptor and resident tensor.
    """

    bank_schema = ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale")

    def __init__(
        self,
        folder: str,
        *,
        num_layers: int,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        resident_bytes: int,
        pin_memory: bool = True,
    ) -> None:
        if resident_bytes <= 0:
            raise ValueError("resident_bytes must be positive")
        self.num_layers = num_layers
        self.num_experts = num_experts
        self._hidden = hidden_size
        self._intermediate = intermediate_size
        self._resident_limit = resident_bytes
        self._pin_memory = pin_memory
        self._lock = threading.RLock()
        self._closed = False
        self._fds: dict[str, int] = {}
        self._ranges = self._index(folder)
        self._cache: collections.OrderedDict[tuple[int, int], dict[str, torch.Tensor]] = (
            collections.OrderedDict()
        )
        self._resident_bytes = 0
        self._row_shapes = {
            "gate_up_packed": (2 * intermediate_size, hidden_size // 2),
            "gate_up_scale": (2 * intermediate_size, hidden_size // 32),
            "down_packed": (hidden_size, intermediate_size // 2),
            "down_scale": (hidden_size, intermediate_size // 32),
        }
        self._row_dtypes = {
            "gate_up_packed": torch.uint8,
            "gate_up_scale": torch.float8_e8m0fnu,
            "down_packed": torch.uint8,
            "down_scale": torch.float8_e8m0fnu,
        }

    @property
    def resident_bytes(self) -> int:
        with self._lock:
            return self._resident_bytes

    @property
    def resident_limit(self) -> int:
        return self._resident_limit

    @property
    def row_shapes(self) -> dict[str, tuple[int, ...]]:
        return dict(self._row_shapes)

    @property
    def row_dtypes(self) -> dict[str, torch.dtype]:
        return dict(self._row_dtypes)

    @property
    def open_descriptors(self) -> int:
        with self._lock:
            return len(self._fds)

    def get(self, layer: int, expert: int) -> dict[str, torch.Tensor]:
        """Return one exact four-bank row, reading and promoting it on a miss."""
        if not 0 <= layer < self.num_layers or not 0 <= expert < self.num_experts:
            raise IndexError(f"expert row out of range: layer={layer}, expert={expert}")
        key = (layer, expert)
        with self._lock:
            self._require_open()
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit
            row = self._read_row(layer, expert)
            size = sum(t.nbytes for t in row.values())
            if size > self._resident_limit:
                raise ValueError(
                    f"one expert row needs {size} bytes, resident tier is {self._resident_limit}"
                )
            while self._cache and self._resident_bytes + size > self._resident_limit:
                _, evicted = self._cache.popitem(last=False)
                self._resident_bytes -= sum(t.nbytes for t in evicted.values())
            self._cache[key] = row
            self._resident_bytes += size
            return row

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._cache.clear()
            self._resident_bytes = 0
            for fd in self._fds.values():
                os.close(fd)
            self._fds.clear()
            self._closed = True

    def __enter__(self) -> "Dsfp4SafetensorSource":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("expert source is closed")

    def _index(self, folder: str) -> dict[str, _TensorRange]:
        index_path = os.path.join(folder, "model.safetensors.index.json")
        try:
            with open(index_path, encoding="utf-8") as stream:
                weight_map = json.load(stream)["weight_map"]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid safetensor index {index_path}: {exc}") from exc
        wanted = {}
        for layer in range(self.num_layers):
            for expert in range(self.num_experts):
                for projection in ("w1", "w2", "w3"):
                    for kind in ("weight", "scale"):
                        name = f"layers.{layer}.ffn.experts.{expert}.{projection}.{kind}"
                        shard = weight_map.get(name)
                        if shard is None:
                            raise ValueError(f"safetensor index is missing {name}")
                        wanted[name] = os.path.join(folder, shard)
        headers: dict[str, dict] = {}
        data_starts: dict[str, int] = {}
        file_sizes: dict[str, int] = {}
        for path in sorted(set(wanted.values())):
            try:
                with open(path, "rb") as stream:
                    prefix = stream.read(8)
                    if len(prefix) != 8:
                        raise ValueError("truncated header length")
                    header_len = struct.unpack("<Q", prefix)[0]
                    header = stream.read(header_len)
                    if len(header) != header_len:
                        raise ValueError("truncated header")
                    headers[path] = json.loads(header)
                data_starts[path] = 8 + header_len
                file_sizes[path] = os.path.getsize(path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid safetensor shard {path}: {exc}") from exc
        result = {}
        for name, path in wanted.items():
            try:
                meta = headers[path][name]
                dtype = _DTYPE[meta["dtype"]]
                begin, end = meta["data_offsets"]
                shape = tuple(meta["shape"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid tensor metadata for {name}") from exc
            offset = data_starts[path] + begin
            length = end - begin
            elements = 1
            for dimension in shape:
                if not isinstance(dimension, int) or dimension < 0:
                    raise ValueError(f"invalid shape for {name}: {shape}")
                elements *= dimension
            if begin < 0 or end < begin or offset + length > file_sizes[path] or length != elements:
                raise ValueError(f"invalid data range for {name}: {begin}:{end}")
            result[name] = _TensorRange(path, offset, length, shape, dtype)
        return result

    def _read_tensor(self, name: str) -> torch.Tensor:
        entry = self._ranges[name]
        fd = self._fds.get(entry.path)
        if fd is None:
            fd = os.open(entry.path, os.O_RDONLY)
            self._fds[entry.path] = fd
        data = os.pread(fd, entry.length, entry.offset)
        if len(data) != entry.length:
            raise OSError(f"short read for {name}: got {len(data)}, wanted {entry.length}")
        tensor = torch.frombuffer(bytearray(data), dtype=entry.dtype).reshape(entry.shape)
        if self._pin_memory:
            tensor = tensor.pin_memory()
        return tensor

    def _read_row(self, layer: int, expert: int) -> dict[str, torch.Tensor]:
        prefix = f"layers.{layer}.ffn.experts.{expert}"
        w1 = self._read_tensor(f"{prefix}.w1.weight").view(torch.uint8)
        w3 = self._read_tensor(f"{prefix}.w3.weight").view(torch.uint8)
        s1 = self._read_tensor(f"{prefix}.w1.scale")
        s3 = self._read_tensor(f"{prefix}.w3.scale")
        row = {
            "gate_up_packed": torch.cat((w1, w3)),
            "gate_up_scale": torch.cat((s1, s3)),
            "down_packed": self._read_tensor(f"{prefix}.w2.weight").view(torch.uint8),
            "down_scale": self._read_tensor(f"{prefix}.w2.scale"),
        }
        for name, tensor in row.items():
            if tensor.shape != self._row_shapes[name] or tensor.dtype != self._row_dtypes[name]:
                raise ValueError(
                    f"{prefix} produced invalid {name}: {tensor.shape}/{tensor.dtype}, "
                    f"expected {self._row_shapes[name]}/{self._row_dtypes[name]}"
                )
            if self._pin_memory and not tensor.is_pinned():
                row[name] = tensor.pin_memory()
        return row
