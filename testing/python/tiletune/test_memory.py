"""Memory ordering retains uncertain schedules without inventing compute costs."""

import json

import pytest
import tilelang.language as T

from tilelang.tiletune import analyze_prim_func, TileTuneConfig
from tiletune_core.memory import classify_bound, score_memory
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


@pytest.mark.parametrize("kernel_kind", ["gemm", "attention"])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_memory_mode_skips_timing_occupancy_and_family_policies(monkeypatch, tmp_path, kernel_kind, diagnostics):
    from test_modules import attention

    def fail(*args, **kwargs):
        pytest.fail("memory mode must not invoke a timing/occupancy/specialization model")

    monkeypatch.setattr("tilelang.tiletune.pipeline.analyze_pipeline", fail)
    monkeypatch.setattr("tilelang.tiletune.occupancy.analyze_waves", fail)
    monkeypatch.setattr("tilelang.tiletune.engine.predict_warp_specialization", fail)
    monkeypatch.setattr("tilelang.tiletune.engine.select_specialization", fail)
    monkeypatch.setattr("tilelang.tiletune.families.base.KernelSpecialization.__init__", fail)
    func = gemm(stages=3) if kernel_kind == "gemm" else attention()
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


def test_bound_aware_metric_stays_lightweight(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("bound-aware memory mode invoked a timing/occupancy/specialization model")

    monkeypatch.setattr("tilelang.tiletune.pipeline.analyze_pipeline", forbidden)
    monkeypatch.setattr("tilelang.tiletune.occupancy.analyze_waves", forbidden)
    monkeypatch.setattr("tilelang.tiletune.engine.predict_warp_specialization", forbidden)
    monkeypatch.setattr("tilelang.tiletune.engine.select_specialization", forbidden)
    monkeypatch.setattr("tilelang.tiletune.families.base.KernelSpecialization.__init__", forbidden)
    result = analyze_prim_func(
        gemm(stages=2),
        {"ranking_metric": "bound_aware"},
        target={"kind": "cuda", "arch": "sm_80"},
        device_limits=LIMITS,
    )
    assert result["modules"]["bound"]["precision"] == "estimate"
    assert result["modules"]["pipeline_overlap"]["precision"] == "disabled"
    assert result["modules"]["waves"]["precision"] == "disabled"


def test_bound_aware_key_drops_the_duplicate_launch_wave_component():
    access = [dict(operation=0, bytes=64, visits=4)]
    with_waves = score_memory(access, 132, 132)
    without_waves = score_memory(access, 132, 132, include_launch_waves=False)
    assert with_waves["launch_waves_component"] is True
    assert without_waves["launch_waves_component"] is False
    # The three-level key is exactly (U, -depth, access-waves).
    base = without_waves["adjusted_logical_byte_waves"] + 1
    depth_inv = 65535 - without_waves["pipeline_depth"]
    expected = (without_waves["adjusted_logical_byte_waves"] * 65536 + depth_inv) * base
    expected += without_waves["logical_memory_access_waves"]
    assert without_waves["score"] == expected
    assert without_waves["score"] != with_waves["score"]


def test_bound_classifier_uses_the_ridge_point():
    assert classify_bound(4096, 8, 200) == "compute"
    assert classify_bound(400, 8, 200) == "memory"
    assert classify_bound(0, 8, 200) == "memory"
    assert classify_bound(None, 8, 200) is None
    assert classify_bound(4096, None, 200) is None
    assert classify_bound(4096, 0, 200) is None
    with pytest.raises(ValueError):
        classify_bound(-1, 8, 200)
    with pytest.raises(ValueError):
        classify_bound(4096, 8, 0)


def test_occupancy_penalty_scales_only_byte_waves():
    access = [dict(operation=0, bytes=64, visits=4)]
    neutral = score_memory(access, 132, 132)
    penalized = score_memory(access, 132, 132, occupancy_penalty=4)
    assert penalized["score"] > neutral["score"]
    assert penalized["tie_break_score"] == neutral["tie_break_score"]
    assert penalized["occupancy_penalty"] == 4
    assert penalized["adjusted_logical_byte_waves"] == 4 * neutral["adjusted_logical_byte_waves"]
    with pytest.raises(ValueError, match="occupancy_penalty"):
        score_memory(access, 132, 132, occupancy_penalty=0)


def test_bound_aware_metric_reports_and_applies_the_roofline_split():
    ampere = {"kind": "cuda", "arch": "sm_80"}
    result = analyze_prim_func(
        gemm(stages=2), {"ranking_metric": "bound_aware", "memory_diagnostics": True}, target=ampere, device_limits=LIMITS
    )
    bound = result["modules"]["bound"]
    assert bound["bound"] in ("compute", "memory")
    assert bound["ridge_flops_per_byte"] == 200.0
    assert result["tile_cost"]["ranking_metric"] == "bound_aware"
    memory = result["modules"]["memory_traffic"]
    assert result["modules"]["ranking"]["launch_waves_component"] is False
    expected = score_memory(
        memory["accesses"],
        memory["grid_blocks"],
        LIMITS["sm_count"],
        memory["pipeline_depth"],
        occupancy_penalty=bound["occupancy_penalty"],
        include_launch_waves=False,
    )
    assert result["tile_cost"]["score"] == expected["score"]


def test_bound_aware_keeps_a_pure_copy_on_memory_order():
    @T.prim_func
    def kernel(A: T.Tensor((4096,), "float16"), B: T.Tensor((4096,), "float16")):
        with T.Kernel(1, threads=128):
            tile = T.alloc_shared((96,), "float16")
            for k in T.Pipelined(T.ceildiv(4096, 96), num_stages=2):
                T.copy(A[k * 96], tile)
                T.copy(tile, B[k * 96])

    result = analyze_prim_func(kernel, {"ranking_metric": "bound_aware"}, target={"kind": "cuda", "arch": "sm_80"}, device_limits=LIMITS)
    assert result["modules"]["bound"]["bound"] == "memory"
    assert result["modules"]["bound"]["occupancy_penalty"] == 1


def test_wave_rounding_memory_ties_and_measurement_independence():
    fine = score_memory([dict(operation=0, bytes=64, visits=4)], 133, 132)
    coarse = score_memory([dict(operation=0, bytes=128, visits=2)], 133, 132)
    assert fine["logical_byte_waves"] == coarse["logical_byte_waves"] == 512
    assert fine["score"] == coarse["score"]
    records = [
        dict(index=2, tile_cost=dict(fine, ranking_metric="memory")),
        dict(index=7, tile_cost=dict(coarse, ranking_metric="memory")),
        dict(index=4, tile_cost=dict(coarse, ranking_metric="memory")),
    ]
    expected = rank_records(records)
    assert [row["index"] for row in expected] == [4, 7, 2]
    assert all(row["rank"] == row["tie_last_rank"] == 3 and row["tie_first_rank"] == 1 for row in expected)
    assert [row["position"] for row in expected] == [1, 2, 3]
    assert select_top_k(expected, 1) == [4, 7, 2]
    for record in records:
        record.update(latency_ms=-record["index"], winner=True, compiler_resources={"registers": 255})
    assert rank_records(records) == expected


@pytest.mark.parametrize("field,value", [("bytes", -1), ("bytes", True), ("visits", 1.5)])
def test_invalid_resolved_memory_facts_raise(field, value):
    access = dict(operation=0, bytes=16, visits=2)
    access[field] = value
    with pytest.raises(ValueError):
        score_memory([access], 1, 132)


def test_pipeline_depth_only_breaks_equal_byte_work():
    access = [dict(operation=0, bytes=64, visits=4)]
    shallow = score_memory(access, 132, 132, pipeline_depth=1)
    deep = score_memory(access, 132, 132, pipeline_depth=6)
    more_bytes = score_memory([dict(operation=0, bytes=257, visits=1)], 132, 132, pipeline_depth=6)
    assert deep["score"] < shallow["score"] < more_bytes["score"]
    assert deep["logical_byte_waves"] == shallow["logical_byte_waves"] == 256


@pytest.mark.parametrize("depth", [0, -1, True, 1.5, 65536])
def test_invalid_pipeline_depth(depth):
    with pytest.raises(ValueError, match="pipeline_depth"):
        score_memory([], 1, 132, depth)


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


@pytest.mark.parametrize("kernel_kind", ["gemm"])
def test_lean_memory_does_not_run_diagnostic_analyses(monkeypatch, tmp_path, kernel_kind):
    def forbidden(*args, **kwargs):
        pytest.fail("lean memory analysis ran a diagnostic-only stage")

    monkeypatch.setattr("tilelang.tiletune.src.collector._Collector._collect_dependencies", forbidden)
    monkeypatch.setattr("tilelang.tiletune.engine.analyze_live_tiles", forbidden)
    monkeypatch.setattr("tilelang.tiletune.engine._propagate_tiles", forbidden)
    func = gemm()
    # No strict register demand policy needs backward propagation here.
    settings = dict(ranking_metric="memory", max_spill_bytes=None, max_local_bytes=None, trace_path=str(tmp_path / "trace.log"))
    result = analyze_prim_func(func, settings, target=TARGET, device_limits=LIMITS)
    assert result["tile_cost"]["score"] > 0
    assert result["tile_propagation"]["resource_demands_computed"] is False
    assert result["pressure"]["tile_liveness"]["peak_registers_per_block_estimate"] is None
    assert result["modules"]["shared_memory"]["shared_memory_bytes_estimate"] is not None
    assert result["modules"]["shared_memory"]["shared_storage_plan"]["precision"] == "disabled"
    assert '"dependencies_collected": false' in (tmp_path / "trace.log").read_text()


def test_estimated_shared_memory_above_device_block_limit_is_deprioritized():
    limits = {**LIMITS, "shared_memory_per_block": 1024}
    result = analyze_prim_func(gemm(stages=3, explicit=True), dict(ranking_metric="memory"), target=TARGET, device_limits=limits)
    decision = result["pressure"]["decision"]
    estimate = result["modules"]["shared_memory"]["shared_memory_bytes_estimate"]
    assert estimate > limits["shared_memory_per_block"]
    assert decision["capacity_reasons"] == [f"estimated shared memory {estimate} exceeds device block limit 1024"]
    assert decision["classification"] == "resource_violation" and decision["status"] == "reject"
    # The arena is estimated, so the block is ranked last rather than refused.
    assert decision["would_reject"] and decision["keep"]
    assert not decision["reasons"] and not decision["physical_reasons"]
    ranking = rank_records([dict(index=0, tile_cost=result["tile_cost"], pre_lowering=decision)])
    assert ranking[0]["tier"] == "pressure_rejected"
    assert select_top_k(ranking, 1, strict_budget=True) == []


@pytest.mark.parametrize("kernel_kind", ["gemm", "attention"])
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
    from test_modules import attention

    func = gemm(stages=3, explicit=True) if kernel_kind == "gemm" else attention()
    before = func.script()
    lean, full = [
        analyze_prim_func(
            func, dict(ranking_metric="memory", memory_diagnostics=diagnostics, **settings), target=TARGET, device_limits=limits
        )
        for diagnostics in (False, True)
    ]
    assert func.script() == before
    for key in ("score", "tie_break_score", "logical_byte_waves", "pipeline_depth", "precision", "unknown"):
        assert lean["tile_cost"][key] == full["tile_cost"][key]
    for key in ("keep", "would_reject", "physical_reasons", "capacity_reasons", "policy_reasons"):
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
