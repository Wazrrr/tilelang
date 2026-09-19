"""Memory ordering retains uncertain schedules without inventing compute costs."""

import json

import pytest
import tilelang.language as T

from tilelang.tiletune import analyze_prim_func, TileTuneConfig
from tiletune_core.memory import score_memory
from tiletune_core.ranking import rank_records, select_top_k
from test_analysis import gemm
from test_cost import LIMITS
from test_modules import TARGET


def test_logical_tail_accesses_do_not_expand_to_whole_tensor():
    @T.prim_func
    def kernel(A: T.Tensor((4096,), "float16"), B: T.Tensor((4096,), "float16")):
        with T.Kernel(1, threads=128):
            tile = T.alloc_shared((96,), "float16")
            for k in T.Pipelined(T.ceildiv(4096, 96), num_stages=2):
                T.copy(A[k * 96], tile)
                T.copy(tile, B[k * 96])

    before = kernel.script()
    result = analyze_prim_func(kernel, dict(ranking_metric="memory"), target=TARGET, device_limits=LIMITS)
    assert kernel.script() == before
    accesses = result["modules"]["memory_traffic"]["accesses"]
    assert [a["bytes"] for a in accesses] == [96 * 2, 96 * 2]
    assert [a["visits"] for a in accesses] == [43, 43]
    assert result["tile_cost"]["score"] == 2 * 96 * 2 * 43
    assert any(d["predecessors"] for d in result["modules"]["memory_traffic"]["dependencies"])


def test_memory_mode_skips_timing_occupancy_and_warp_policy(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        pytest.fail("memory mode must not invoke a timing/occupancy/specialization model")

    monkeypatch.setattr("tilelang.tiletune.pipeline.analyze_pipeline", fail)
    monkeypatch.setattr("tilelang.tiletune.occupancy.analyze_waves", fail)
    monkeypatch.setattr("tilelang.tiletune.engine.predict_warp_specialization", fail)
    monkeypatch.setattr("tilelang.tiletune.engine.select_specialization", fail)
    path = tmp_path / "memory.json"
    result = analyze_prim_func(gemm(stages=3), dict(ranking_metric="memory", facts_path=str(path)), target=TARGET, device_limits=LIMITS)
    facts = json.loads(path.read_text())
    assert facts["backend"] == "memory.v1"
    assert score_memory(facts["accesses"], facts["grid_blocks"], facts["sm_count"])["score"] == result["tile_cost"]["score"]
    assert result["modules"]["pipeline_overlap"]["precision"] == "disabled"
    assert result["modules"]["waves"]["precision"] == "disabled"
    assert result["pressure"]["tile_liveness"]["peak_registers_per_block_estimate"] > 0


def test_soft_register_overflow_does_not_remove_memory_score():
    from examples.flash_attention.example_mha_fwd_bshd import flashattn

    func = flashattn.jit_impl.get_tir(
        batch=1, heads=32, seq_len=4096, dim=128, is_causal=True, block_M=128, block_N=256, num_stages=1, threads=256
    )
    result = analyze_prim_func(
        func,
        dict(ranking_metric="memory", register_cap=200, max_spill_bytes=None, max_local_bytes=None),
        target=TARGET,
        device_limits=LIMITS,
    )
    assert result["pressure"]["register_demand"]["status"] == "exceeds_allowance"
    assert not result["pressure"]["decision"]["would_reject"]
    assert result["tile_cost"]["score"] > 0
    records = [dict(index=0, tile_cost=result["tile_cost"], pre_lowering=result["pressure"]["decision"])]
    assert select_top_k(rank_records(records), 1) == [0]


def test_explicit_resource_policy_still_rejects():
    result = analyze_prim_func(gemm(), dict(ranking_metric="memory", register_cap=1), target=TARGET, device_limits=LIMITS)
    assert result["pressure"]["decision"]["would_reject"]
    records = [dict(index=0, tile_cost=result["tile_cost"], pre_lowering=result["pressure"]["decision"])]
    assert result["tile_cost"]["score"] > 0
    assert select_top_k(rank_records(records), 1) == []


def test_missing_memory_effects_and_device_inputs_remain_unknown():
    @T.prim_func
    def opaque(A: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32):
            T.evaluate(T.call_extern("int32", "opaque_memory_effect", A.data))
            for i in T.Parallel(32):
                A[i] = 0

    result = analyze_prim_func(opaque, dict(ranking_metric="memory"), target=TARGET, device_limits=LIMITS)
    assert result["tile_cost"]["score"] is None
    assert result["tile_cost"]["unknown"]
    result = analyze_prim_func(gemm(), dict(ranking_metric="memory"), target=TARGET)
    assert result["tile_cost"]["score"] is None
    assert "unresolved sm_count" in result["tile_cost"]["unknown"]


def test_wave_rounding_memory_ties_and_measurement_independence():
    # Equal logical bytes, but fewer transfers/dependency events in candidate 7.
    fine = score_memory([dict(operation=0, bytes=64, visits=4)], 133, 132)
    coarse = score_memory([dict(operation=0, bytes=128, visits=2)], 133, 132)
    assert fine["score"] == coarse["score"] == 512
    records = [
        dict(index=2, tile_cost=dict(fine, ranking_metric="memory")),
        dict(index=7, tile_cost=dict(coarse, ranking_metric="memory")),
        dict(index=4, tile_cost=dict(coarse, ranking_metric="memory")),
    ]
    expected = rank_records(records)
    assert [r["index"] for r in expected] == [4, 7, 2]
    assert expected[0]["tie_first_rank"] == 1 and expected[0]["tie_last_rank"] == 2
    assert expected[-1]["tie_first_rank"] == 3
    for r in records:
        r.update(latency_ms=-r["index"], winner=True, compiler_resources={"registers": 255})
    assert rank_records(records) == expected
    assert TileTuneConfig(ranking_metric="memory").to_cache_key_dict() != TileTuneConfig().to_cache_key_dict()


@pytest.mark.parametrize("field,value", [("bytes", -1), ("bytes", True), ("visits", 1.5)])
def test_invalid_resolved_memory_facts_raise(field, value):
    access = dict(operation=0, bytes=16, visits=2)
    access[field] = value
    with pytest.raises(ValueError):
        score_memory([access], 1, 132)
