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
                weight = (torch.arange(torch.tensor(shape).prod()).reshape(shape) + seed).to(torch.uint8).view(torch.int8)
                scale_shape = (shape[0], shape[1] // 16)
                scale = (torch.arange(torch.tensor(scale_shape).prod()).reshape(scale_shape) + seed).to(torch.uint8)
                base = f"layers.{layer}.ffn.experts.{expert}.{projection}"
                tensors[f"{base}.weight"] = (
                    "I8", shape, bytes(weight.view(torch.uint8).flatten().tolist())
                )
                tensors[f"{base}.scale"] = ("F8_E8M0", scale_shape, bytes(scale.flatten().tolist()))
                row[projection] = (weight.view(torch.uint8), scale)
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


class _FakeRowSource:
    bank_schema = ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale")
    num_layers = 1
    num_experts = 3
    row_shapes = {name: (2,) for name in bank_schema}
    row_dtypes = {name: torch.uint8 for name in bank_schema}

    def __init__(self):
        self.closed = False

    def get(self, layer, expert):
        assert layer == 0
        return {
            name: torch.tensor([expert, bank], dtype=torch.uint8)
            for bank, name in enumerate(self.bank_schema)
        }

    def close(self):
        self.closed = True


def test_offload_cache_consumes_rows_in_existing_slot_pool():
    from freetoken.moe.offload_cache import OffloadMoeCache

    source = _FakeRowSource()
    cache = OffloadMoeCache(
        num_layers=1, num_experts=3, cache_size=3,
        device=torch.device("cpu"), quant_format="ds_fp4",
    )
    cache.set_row_source(source)
    cache._pending_src_layer = 0
    cache.num_indices.fill_(2)
    cache.src_indices[:2] = torch.tensor([2, 0], dtype=torch.int32)
    cache.evict_slots[:2] = torch.tensor([1, 2], dtype=torch.int32)
    cache.copy_missing()
    for bank, name in enumerate(source.bank_schema):
        assert cache.bank_caches[name][1].tolist() == [2, bank]
        assert cache.bank_caches[name][2].tolist() == [0, bank]
    assert len(cache.bank_caches) == 4
    cache.close()
    assert source.closed


def test_row_source_rejects_prefill_overlap():
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(
        num_layers=1, num_experts=3, cache_size=6,
        device=torch.device("cpu"), quant_format="ds_fp4", prefill_overlap=True,
    )
    with pytest.raises(ValueError, match="disable-moe-prefill-overlap"):
        cache.set_row_source(_FakeRowSource())


def test_ssd_streaming_disables_cuda_graph_replay():
    from types import SimpleNamespace

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    from freetoken.engine.engine import _adjust_config

    args = SimpleNamespace(window_size=64, max_seq_len=0, max_batch_size=0)
    model = SimpleNamespace(
        dsv4_args=args, single_stream_only=False, has_swa_attention=False,
        has_linear_attention=False, is_moe=True, expert_quant="ds_fp4",
        num_layers=2, num_moe_layers=2, num_experts=4,
        rotary_config=SimpleNamespace(m=4096, max_position=4096), moe_backend="offload",
    )
    config = EngineConfig(
        model_path="/tmp/dsv4", tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16, moe_backend="offload",
        moe_expert_resident_bytes=4096, cuda_graph_max_bs=4,
        cuda_graph_bs=[1, 2, 4], max_running_req=4,
    )
    object.__setattr__(config, "model_config", model)
    _adjust_config(config)
    assert config.cuda_graph_max_bs == 0
    assert config.cuda_graph_bs == []
    assert config.moe_prefill_overlap is False


class _FailSecondRowSource(_FakeRowSource):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.fail = True

    def get(self, layer, expert):
        self.calls += 1
        if self.fail and self.calls == 2:
            raise OSError("synthetic SSD failure")
        return super().get(layer, expert)


def test_copy_failure_invalidates_slot_maps_and_retry_is_exact():
    from freetoken.moe.offload_cache import OffloadMoeCache

    source = _FailSecondRowSource()
    cache = OffloadMoeCache(
        num_layers=1, num_experts=3, cache_size=3,
        device=torch.device("cpu"), quant_format="ds_fp4",
    )
    cache.set_row_source(source)
    cache.slot_for_id[0, :2] = torch.tensor([1, 2], dtype=torch.int32)
    cache.id_of_slot[1:3] = torch.tensor([0, 1], dtype=torch.int32)
    cache._pending_src_layer = 0
    cache.num_indices.fill_(2)
    cache.src_indices[:2] = torch.tensor([0, 1], dtype=torch.int32)
    cache.evict_slots[:2] = torch.tensor([1, 2], dtype=torch.int32)
    with pytest.raises(OSError, match="synthetic SSD failure"):
        cache.copy_missing()
    assert torch.all(cache.slot_for_id == -1)
    assert torch.all(cache.id_of_slot == -1)

    source.fail = False
    cache._pending_src_layer = 0
    cache.num_indices.fill_(2)
    cache.src_indices[:2] = torch.tensor([0, 1], dtype=torch.int32)
    cache.evict_slots[:2] = torch.tensor([1, 2], dtype=torch.int32)
    cache.copy_missing()
    for bank, name in enumerate(source.bank_schema):
        assert cache.bank_caches[name][1].tolist() == [0, bank]
        assert cache.bank_caches[name][2].tolist() == [1, bank]


def test_dense_model_ignores_ssd_setting_without_disabling_graphs():
    from types import SimpleNamespace

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    from freetoken.engine.engine import _adjust_config

    config = EngineConfig(
        model_path="/tmp/dense", tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.float16, moe_expert_resident_bytes=4096,
        cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4],
    )
    object.__setattr__(config, "model_config", SimpleNamespace(
        single_stream_only=False, has_swa_attention=False,
        has_linear_attention=False, is_moe=False, expert_quant="none",
        dsv4_args=None, model_type="dense",
    ))
    _adjust_config(config)
    assert config.moe_expert_resident_bytes == 0
    assert config.cuda_graph_max_bs == 4
    assert config.cuda_graph_bs == [1, 2, 4]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
def test_rocm_row_source_rebuild_decode_miss_and_materialize(tmp_path):
    from freetoken.moe.offload_cache import OffloadMoeCache

    expected = _checkpoint(tmp_path, layers=1, experts=4)
    row_bytes = sum(t.nbytes for t in expected[0, 0].values())
    source = Dsfp4SafetensorSource(
        str(tmp_path), num_layers=1, num_experts=4, hidden_size=64,
        intermediate_size=32, resident_bytes=2 * row_bytes, pin_memory=True,
    )
    cache = OffloadMoeCache(
        num_layers=1, num_experts=4, cache_size=4,
        device=torch.device("cuda"), quant_format="ds_fp4",
    )
    cache.set_row_source(source)
    ids = torch.tensor([[2, 0]], dtype=torch.int32, device="cuda")
    cache.ensure_experts(0, ids)
    cache.copy_missing()
    for routed, slot in zip((2, 0), ids.cpu().flatten().tolist()):
        for name in source.bank_schema:
            assert torch.equal(
                cache.bank_caches[name][slot].cpu().view(torch.uint8),
                expected[0, routed][name].view(torch.uint8),
            )
    assert source.resident_bytes <= source.resident_limit

    cache.rebuild(4)
    ids = torch.tensor([[1]], dtype=torch.int32, device="cuda")
    cache.ensure_experts(0, ids)
    cache.copy_missing()
    for name in source.bank_schema:
        assert torch.equal(
            cache.bank_caches[name][ids.item()].cpu().view(torch.uint8),
            expected[0, 1][name].view(torch.uint8),
        )

    cache.materialize_layer(0)
    cache.copy_missing()
    for expert in range(4):
        for name in source.bank_schema:
            assert torch.equal(
                cache.bank_caches[name][expert].cpu().view(torch.uint8),
                expected[0, expert][name].view(torch.uint8),
            )
    assert source.resident_bytes <= source.resident_limit
    cache.close()
