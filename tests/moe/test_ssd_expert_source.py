from __future__ import annotations

import json
import struct
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from freetoken.moe.ssd_expert_source import Dsfp4SafetensorSource


def _checkpoint(path, *, layers=2, experts=4, hidden=64, intermediate=32):
    tensors = {}
    expected = {}
    for layer in range(layers):
        for expert in range(experts):
            row = {}
            for projection, shape in (
                ("w1", (intermediate, hidden // 2)),
                ("w3", (intermediate, hidden // 2)),
                ("w2", (hidden, intermediate // 2)),
            ):
                seed = layer * 80 + expert * 12 + {"w1": 1, "w2": 3, "w3": 5}[projection]
                weight = (torch.arange(torch.tensor(shape).prod()).reshape(shape) + seed).to(torch.uint8)
                scale_shape = (shape[0], shape[1] // 16)
                scale = (torch.arange(torch.tensor(scale_shape).prod()).reshape(scale_shape) + seed).to(torch.uint8)
                base = f"layers.{layer}.ffn.experts.{expert}.{projection}"
                tensors[f"{base}.weight"] = ("U8", shape, bytes(weight.flatten().tolist()))
                tensors[f"{base}.scale"] = ("F8_E8M0", scale_shape, bytes(scale.flatten().tolist()))
                row[projection] = (weight, scale)
            expected[layer, expert] = {
                "gate_up_packed": torch.cat((row["w1"][0], row["w3"][0])),
                "gate_up_scale": torch.cat((row["w1"][1], row["w3"][1])).view(torch.float8_e8m0fnu),
                "down_packed": row["w2"][0],
                "down_scale": row["w2"][1].view(torch.float8_e8m0fnu),
            }
    header, payload, weight_map = {}, bytearray(), {}
    for name, (dtype, shape, data) in tensors.items():
        begin = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [begin, len(payload)]}
        weight_map[name] = "experts.safetensors"
    encoded = json.dumps(header, separators=(",", ":")).encode()
    (path / "experts.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return expected


def _assert_row(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        assert torch.equal(actual[name].view(torch.uint8), expected[name].view(torch.uint8))


def test_reads_exact_rows_with_bounded_lru_and_cleanup(tmp_path):
    expected = _checkpoint(tmp_path)
    row_bytes = sum(t.nbytes for t in expected[0, 0].values())
    source = Dsfp4SafetensorSource(
        str(tmp_path), num_layers=2, num_experts=4, hidden_size=64,
        intermediate_size=32, resident_bytes=2 * row_bytes, pin_memory=False,
    )
    first = source.get(0, 0)
    _assert_row(first, expected[0, 0])
    assert source.get(0, 0) is first
    _assert_row(source.get(1, 3), expected[1, 3])
    _assert_row(source.get(0, 2), expected[0, 2])
    assert source.resident_bytes == 2 * row_bytes
    assert source.get(0, 0) is not first
    assert source.resident_bytes == 2 * row_bytes
    assert source.open_descriptors == 1
    source.close()
    assert source.open_descriptors == 0
    assert source.resident_bytes == 0
    with pytest.raises(RuntimeError, match="closed"):
        source.get(0, 0)


def test_concurrent_reads_are_exact(tmp_path):
    expected = _checkpoint(tmp_path)
    row_bytes = sum(t.nbytes for t in expected[0, 0].values())
    with Dsfp4SafetensorSource(
        str(tmp_path), num_layers=2, num_experts=4, hidden_size=64,
        intermediate_size=32, resident_bytes=3 * row_bytes, pin_memory=False,
    ) as source:
        keys = [(i % 2, i % 4) for i in range(40)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            rows = list(pool.map(lambda key: source.get(*key), keys))
        for key, row in zip(keys, rows):
            _assert_row(row, expected[key])
        assert source.resident_bytes <= source.resident_limit


def test_rejects_ranges_and_malformed_index(tmp_path):
    expected = _checkpoint(tmp_path)
    row_bytes = sum(t.nbytes for t in expected[0, 0].values())
    source = Dsfp4SafetensorSource(
        str(tmp_path), num_layers=2, num_experts=4, hidden_size=64,
        intermediate_size=32, resident_bytes=row_bytes, pin_memory=False,
    )
    with pytest.raises(IndexError, match="out of range"):
        source.get(2, 0)
    source.close()

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    index["weight_map"].pop(next(iter(index["weight_map"])))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="is missing"):
        Dsfp4SafetensorSource(
            str(tmp_path), num_layers=2, num_experts=4, hidden_size=64,
            intermediate_size=32, resident_bytes=row_bytes, pin_memory=False,
        )

    _checkpoint(tmp_path)
    shard = tmp_path / "experts.safetensors"
    shard.write_bytes(shard.read_bytes()[:-1])
    with pytest.raises(ValueError, match="invalid data range"):
        Dsfp4SafetensorSource(
            str(tmp_path), num_layers=2, num_experts=4, hidden_size=64,
            intermediate_size=32, resident_bytes=row_bytes, pin_memory=False,
        )


def test_rejects_tensor_larger_than_resident_tier(tmp_path):
    expected = _checkpoint(tmp_path)
    row_bytes = sum(t.nbytes for t in expected[0, 0].values())
    source = Dsfp4SafetensorSource(
        str(tmp_path), num_layers=2, num_experts=4, hidden_size=64,
        intermediate_size=32, resident_bytes=row_bytes - 1, pin_memory=False,
    )
    with pytest.raises(ValueError, match="one expert row"):
        source.get(0, 0)
    source.close()
