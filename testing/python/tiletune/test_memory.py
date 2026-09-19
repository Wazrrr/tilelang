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
    result = analyze_prim_func(kernel, dict(ranking_metric="memory", memory_diagnostics=True), target=TARGET, device_limits=LIMITS)
    assert kernel.script() == before
    accesses = result["modules"]["memory_traffic"]["accesses"]
    assert [a["bytes"] for a in accesses] == [96 * 2, 96 * 2]
    assert [a["visits"] for a in accesses] == [43, 43]
    assert result["tile_cost"]["logical_byte_waves"] == 2 * 96 * 2 * 43
    assert result["tile_cost"]["pipeline_depth"] == 2
    assert any(d["predecessors"] for d in result["modules"]["memory_traffic"]["dependencies"])


@pytest.mark.parametrize("kernel_kind", ["gemm", "attention", "softmax"])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_memory_mode_skips_timing_occupancy_and_family_policies(monkeypatch, tmp_path, kernel_kind, diagnostics):
    from experiments.softmax.analyze import softmax
    from test_modules import attention

    def fail(*args, **kwargs):
        pytest.fail("memory mode must not invoke a timing/occupancy/specialization model")

    monkeypatch.setattr("tilelang.tiletune.pipeline.analyze_pipeline", fail)
    monkeypatch.setattr("tilelang.tiletune.occupancy.analyze_waves", fail)
    monkeypatch.setattr("tilelang.tiletune.engine.predict_warp_specialization", fail)
    monkeypatch.setattr("tilelang.tiletune.engine.select_specialization", fail)
    monkeypatch.setattr("tilelang.tiletune.families.base.KernelSpecialization.__init__", fail)
    func = gemm(stages=3) if kernel_kind == "gemm" else attention() if kernel_kind == "attention" else softmax(257, 1000, 2, 128)
    path = tmp_path / "memory.json"
    result = analyze_prim_func(
        func, dict(ranking_metric="memory", memory_diagnostics=diagnostics, facts_path=str(path)), target=TARGET, device_limits=LIMITS
    )
    facts = json.loads(path.read_text())
    assert facts["backend"] == "memory.v3"
    assert (
        score_memory(facts["accesses"], facts["grid_blocks"], facts["sm_count"], facts["pipeline_depth"])["score"]
        == result["tile_cost"]["score"]
    )
    assert result["modules"]["pipeline_overlap"]["precision"] == "disabled"
    assert result["modules"]["waves"]["precision"] == "disabled"
    if diagnostics:
        assert result["pressure"]["tile_liveness"]["peak_registers_per_block_estimate"] > 0
        assert facts["dependencies"]
    else:
        assert result["pressure"]["tile_liveness"]["precision"] == "disabled"
        assert facts["dependencies"] is None
    assert result["specialization"]["name"] == "generic"
    assert not result["specialization"]["roles"]


@pytest.mark.parametrize(
    "settings", [dict(specialization="gemm"), dict(specialization="attention"), dict(attention_spill_budget_registers_per_thread=32)]
)
def test_memory_mode_rejects_kernel_family_hints(settings):
    with pytest.raises(ValueError, match="kernel-family independent"):
        TileTuneConfig(ranking_metric="memory", **settings)


@pytest.mark.parametrize("target", [TARGET, {"kind": "hip", "mcpu": "gfx950"}])
def test_memory_model_uses_backend_inputs_without_kernel_family_rules(target):
    @T.prim_func
    def copy(A: T.Tensor((125, 8), "float32"), B: T.Tensor((125, 8), "float32")):
        with T.Kernel(125, threads=128) as block:
            tile = T.alloc_fragment((8,), "float32")
            T.copy(A[block, :], tile)
            T.copy(tile, B[block, :])

    # Synthetic unit counts test the input contract, not either device's speed.
    for units, expected in ((132, 64), (64, 128)):
        result = analyze_prim_func(copy, dict(ranking_metric="memory"), target=target, device_limits={"sm_count": units})
        assert result["tile_cost"]["logical_byte_waves"] == expected
        assert result["specialization"]["name"] == "generic"
        assert result["pressure"]["target_model"]["kind"] == target["kind"]


def test_soft_register_overflow_does_not_remove_memory_score():
    from examples.flash_attention.example_mha_fwd_bshd import flashattn

    func = flashattn.jit_impl.get_tir(
        batch=1, heads=32, seq_len=4096, dim=128, is_causal=True, block_M=128, block_N=256, num_stages=1, threads=256
    )
    result = analyze_prim_func(
        func,
        dict(ranking_metric="memory", memory_diagnostics=True, register_cap=200, max_spill_bytes=None, max_local_bytes=None),
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
    assert fine["logical_byte_waves"] == coarse["logical_byte_waves"] == 512
    assert coarse["score"] < fine["score"]
    records = [
        dict(index=2, tile_cost=dict(fine, ranking_metric="memory")),
        dict(index=7, tile_cost=dict(coarse, ranking_metric="memory")),
        dict(index=4, tile_cost=dict(coarse, ranking_metric="memory")),
    ]
    expected = rank_records(records)
    assert [r["index"] for r in expected] == [4, 7, 2]
    assert [r["rank"] for r in expected] == [2, 2, 3]
    assert [r["position"] for r in expected] == [1, 2, 3]
    assert select_top_k(expected, 1) == [4, 7]
    assert select_top_k(expected, 1, strict_budget=True) == []
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


def test_exact_score_preserves_lexicographic_order_above_float_precision():
    # Exhaust both extreme request counts and depths around adjacent byte bands.
    scores = []
    for size in (2**30, 2**30 + 1):
        for events in (1, size):
            for depth in (65535, 1):
                accesses = [dict(operation=0, bytes=size, visits=1)] if events == 1 else [dict(operation=0, bytes=1, visits=size)]
                result = score_memory(accesses, 1, 132, depth)
                assert type(result["score"]) is int and result["score"] > 2**53
                scores.append(result["score"])
    assert scores == sorted(set(scores))


@pytest.mark.parametrize("depth", [0, -1, True, 1.5, 65536])
def test_invalid_pipeline_depth(depth):
    with pytest.raises(ValueError, match="pipeline_depth"):
        score_memory([], 1, 132, depth)


def test_softmax_memory_score_cannot_distinguish_equal_memory_work(monkeypatch):
    from experiments.softmax.analyze import softmax

    def forbidden(*args, **kwargs):
        pytest.fail("a standalone PrimFunc must not require a kernel-family helper")

    monkeypatch.setattr("tilelang.tiletune.engine.select_specialization", forbidden)
    monkeypatch.setattr("tilelang.tiletune.families.base.KernelSpecialization.__init__", forbidden)
    records = []
    for block_rows in (1, 2, 4, 8):
        for threads in (128, 256):
            func = softmax(4096, 4096, block_rows, threads)
            result = analyze_prim_func(func, {"ranking_metric": "memory"}, target=TARGET, device_limits={"sm_count": 132})
            assert result["tile_cost"]["score"] is not None
            assert not result["tile_cost"]["unknown"]
            records.append(dict(index=len(records), tile_cost=result["tile_cost"], pre_lowering=result["pressure"]["decision"]))
    ranking = rank_records(records)
    assert len({row["score"] for row in ranking}) == 1
    assert all(row["tier"] == "eligible" and row["rank"] == 8 for row in ranking)
    # Successful analysis does not justify splitting a tie to fill alpha=0.5.
    assert select_top_k(ranking, 4, strict_budget=True) == []


def test_local_compute_and_liveness_are_not_memory_score_terms():
    def build(extra_compute):
        @T.prim_func
        def pointwise(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
            with T.Kernel(1, threads=128):
                values = T.alloc_fragment((128,), "float32")
                T.copy(A, values)
                if extra_compute:
                    scratch = T.alloc_fragment((128,), "float32")
                    for i in T.Parallel(128):
                        scratch[i] = T.exp(values[i])
                    for i in T.Parallel(128):
                        values[i] = scratch[i] + values[i]
                T.copy(values, B)

        return pointwise

    baseline, extra = [
        analyze_prim_func(
            build(extra_compute), {"ranking_metric": "memory", "memory_diagnostics": True}, target=TARGET, device_limits={"sm_count": 132}
        )
        for extra_compute in (False, True)
    ]
    assert len(extra["tile_propagation"]["operations"]) > len(baseline["tile_propagation"]["operations"])
    assert (
        extra["pressure"]["tile_liveness"]["peak_registers_per_block_estimate"]
        > baseline["pressure"]["tile_liveness"]["peak_registers_per_block_estimate"]
    )
    assert extra["tile_cost"]["score"] == baseline["tile_cost"]["score"]


@pytest.mark.parametrize("kernel_kind", ["gemm", "softmax"])
def test_lean_memory_does_not_run_diagnostic_analyses(monkeypatch, tmp_path, kernel_kind):
    from experiments.softmax.analyze import softmax

    def forbidden(*args, **kwargs):
        pytest.fail("lean memory analysis ran a diagnostic-only stage")

    monkeypatch.setattr("tilelang.tiletune.src.collector._Collector._collect_dependencies", forbidden)
    monkeypatch.setattr("tilelang.tiletune.engine.analyze_live_tiles", forbidden)
    monkeypatch.setattr("tilelang.tiletune.shared_memory.analyze_shared_memory", forbidden)
    monkeypatch.setattr("tilelang.tiletune.engine._propagate_tiles", forbidden)
    func = gemm() if kernel_kind == "gemm" else softmax(257, 1000, 2, 128)
    # No strict register demand policy needs backward propagation here.
    settings = dict(ranking_metric="memory", max_spill_bytes=None, max_local_bytes=None, trace_path=str(tmp_path / "trace.log"))
    result = analyze_prim_func(func, settings, target=TARGET, device_limits=LIMITS)
    assert result["tile_cost"]["score"] > 0
    assert result["tile_propagation"]["resource_demands_computed"] is False
    assert result["pressure"]["tile_liveness"]["peak_registers_per_block_estimate"] is None
    assert result["modules"]["shared_memory"]["shared_storage_plan"]["precision"] == "disabled"
    assert '"dependencies_collected": false' in (tmp_path / "trace.log").read_text()


@pytest.mark.parametrize("kernel_kind", ["gemm", "attention", "softmax"])
@pytest.mark.parametrize(
    "settings,limits",
    [
        ({}, LIMITS),
        ({"register_cap": 1}, LIMITS),
        ({"register_cap": 1, "mode": "report_only"}, LIMITS),
        ({"max_spill_bytes": None, "max_local_bytes": None}, LIMITS),
        ({}, {**LIMITS, "max_threads_per_block": 64}),
    ],
)
def test_memory_diagnostics_preserve_scores_and_resource_rejections(kernel_kind, settings, limits):
    from experiments.softmax.analyze import softmax
    from test_modules import attention

    func = (
        gemm(stages=3, explicit=True)
        if kernel_kind == "gemm"
        else attention()
        if kernel_kind == "attention"
        else softmax(257, 1000, 2, 128)
    )
    before = func.script()
    lean, full = [
        analyze_prim_func(
            func, dict(ranking_metric="memory", memory_diagnostics=diagnostics, **settings), target=TARGET, device_limits=limits
        )
        for diagnostics in (False, True)
    ]
    assert func.script() == before
    for key in ("score", "logical_byte_waves", "logical_memory_access_waves", "pipeline_depth", "precision", "unknown"):
        assert lean["tile_cost"][key] == full["tile_cost"][key]
    for key in ("keep", "would_reject", "physical_reasons", "policy_reasons"):
        assert lean["pressure"]["decision"][key] == full["pressure"]["decision"][key]
    assert full["tile_propagation"]["operations"]
    assert lean["tile_propagation"]["precision"] == "disabled"


def test_memory_diagnostics_is_explicit_and_cache_distinct():
    lean = TileTuneConfig(ranking_metric="memory")
    full = TileTuneConfig(ranking_metric="memory", memory_diagnostics=True)
    assert not lean.memory_diagnostics
    assert lean.to_cache_key_dict() != full.to_cache_key_dict()
    for value in (None, 0, 1, "true"):
        with pytest.raises(ValueError, match="memory_diagnostics must be a bool"):
            TileTuneConfig(memory_diagnostics=value)
