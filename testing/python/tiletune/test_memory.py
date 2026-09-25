"""Memory ordering retains uncertain schedules without inventing compute costs."""

import json

import pytest
import tilelang.language as T

from tilelang.tiletune import analyze_prim_func, TileTuneConfig
from tiletune_core.memory import classify_bound, score_memory
from tiletune_core.ranking import rank_records, select_top_k
from test_analysis import gemm
from test_cost import LIMITS
from test_modules import TARGET, attention


def test_logical_tail_accesses_do_not_expand_to_whole_tensor():
    @T.prim_func
    def kernel(A: T.Tensor((4096,), "float16"), B: T.Tensor((4096,), "float16")):
        with T.Kernel(1, threads=128):
            tile = T.alloc_shared((96,), "float16")
            for k in T.Pipelined(T.ceildiv(4096, 96), num_stages=2):
                T.copy(A[k * 96], tile)
                T.copy(tile, B[k * 96])

    before = kernel.script()
    result = analyze_prim_func(
        kernel, dict(ranking_metric="memory", memory_diagnostics=True), target=TARGET, device_limits=LIMITS
    )
    assert kernel.script() == before
    accesses = result["modules"]["memory_traffic"]["accesses"]
    assert [access["bytes"] for access in accesses] == [96 * 2, 96 * 2]
    assert [access["visits"] for access in accesses] == [43, 43]
    assert result["tile_cost"]["logical_byte_waves"] == 2 * 96 * 2 * 43
    assert result["tile_cost"]["pipeline_depth"] == 2
    assert any(item["predecessors"] for item in result["modules"]["memory_traffic"]["dependencies"])


@pytest.mark.parametrize("kernel_kind", ["gemm", "attention"])
def test_memory_mode_skips_timing_occupancy_and_family_policies(monkeypatch, tmp_path, kernel_kind):
    def fail(*args, **kwargs):
        pytest.fail("memory mode must not invoke a timing, occupancy, or specialization model")

    monkeypatch.setattr("tilelang.tiletune.pipeline.analyze_pipeline", fail)
    monkeypatch.setattr("tilelang.tiletune.occupancy.analyze_waves", fail)
    monkeypatch.setattr("tilelang.tiletune.engine.predict_warp_specialization", fail)
    monkeypatch.setattr("tilelang.tiletune.engine.select_specialization", fail)
    monkeypatch.setattr("tilelang.tiletune.families.base.KernelSpecialization.__init__", fail)
    func = gemm(stages=3) if kernel_kind == "gemm" else attention()
    path = tmp_path / "memory.json"
    result = analyze_prim_func(
        func,
        dict(ranking_metric="memory", memory_diagnostics=True, facts_path=str(path)),
        target=TARGET,
        device_limits=LIMITS,
    )
    facts = json.loads(path.read_text())
    assert facts["backend"] == "memory.v2"
    assert score_memory(facts["accesses"], facts["grid_blocks"], facts["sm_count"], facts["pipeline_depth"])["score"] == result[
        "tile_cost"
    ]["score"]
    assert result["modules"]["pipeline_overlap"]["precision"] == "disabled"
    assert result["modules"]["waves"]["precision"] == "disabled"
    assert result["pressure"]["tile_liveness"]["peak_registers_per_block_estimate"] > 0
    assert result["specialization"]["name"] == "generic"
    assert not result["specialization"]["roles"]


def test_new_softmax_prim_func_is_analyzed_directly(monkeypatch, tmp_path):
    """No softmax recognizer or analysis helper participates in this path."""
    rows, columns, block_rows = 257, 1000, 2
    block_columns = 1 << (columns - 1).bit_length()

    @T.prim_func
    def softmax(A: T.Tensor((rows, columns), "float16"), B: T.Tensor((rows, columns), "float16")):
        with T.Kernel(T.ceildiv(rows, block_rows), threads=128) as block:
            values = T.alloc_fragment((block_rows, block_columns), "float32")
            row_max = T.alloc_fragment((block_rows,), "float32")
            row_sum = T.alloc_fragment((block_rows,), "float32")
            for i, j in T.Parallel(block_rows, block_columns):
                values[i, j] = T.if_then_else(
                    (block * block_rows + i < rows) & (j < columns),
                    T.cast(A[block * block_rows + i, j], "float32"),
                    -T.infinity("float32"),
                )
            T.reduce_max(values, row_max, dim=1, clear=True)
            for i, j in T.Parallel(block_rows, block_columns):
                values[i, j] = T.exp(values[i, j] - row_max[i])
            T.reduce_sum(values, row_sum, dim=1, clear=True)
            for i, j in T.Parallel(block_rows, block_columns):
                if (block * block_rows + i < rows) & (j < columns):
                    B[block * block_rows + i, j] = values[i, j] / row_sum[i]

    def fail(*args, **kwargs):
        pytest.fail("direct memory analysis must not invoke a kernel-family, timing, or occupancy model")

    monkeypatch.setattr("tilelang.tiletune.pipeline.analyze_pipeline", fail)
    monkeypatch.setattr("tilelang.tiletune.occupancy.analyze_waves", fail)
    monkeypatch.setattr("tilelang.tiletune.engine.predict_warp_specialization", fail)
    monkeypatch.setattr("tilelang.tiletune.engine.select_specialization", fail)
    monkeypatch.setattr("tilelang.tiletune.families.base.KernelSpecialization.__init__", fail)

    before = softmax.script()
    path = tmp_path / "softmax-memory.json"
    result = analyze_prim_func(
        softmax,
        dict(ranking_metric="memory", memory_diagnostics=True, facts_path=str(path)),
        target={"kind": "cuda", "arch": "sm_100a"},
        device_limits={**LIMITS, "sm_count": 148},
    )
    facts = json.loads(path.read_text())
    operations = result["tile_propagation"]["operations"]

    assert softmax.script() == before
    assert result["specialization"]["name"] == "generic"
    assert result["specialization"]["roles"] == {}
    assert sum(operation["kind"] == "reduce" for operation in operations) == 2
    assert any(operation["dependencies"] for operation in operations)
    assert result["tile_cost"]["score"] is not None
    assert not result["tile_cost"]["unknown"]
    assert score_memory(
        facts["accesses"], facts["grid_blocks"], facts["sm_count"], facts["pipeline_depth"]
    )["score"] == result["tile_cost"]["score"]


@pytest.mark.parametrize(
    "settings", [dict(specialization="gemm"), dict(specialization="attention"), dict(attention_spill_budget_registers_per_thread=32)]
)
@pytest.mark.parametrize("metric", ["memory", "bound_aware"])
def test_memory_mode_rejects_kernel_family_hints(settings, metric):
    with pytest.raises(ValueError, match="kernel-family independent"):
        TileTuneConfig(ranking_metric=metric, **settings)


def test_lean_memory_mode_skips_score_independent_analysis(monkeypatch):
    func = attention()
    settings = dict(ranking_metric="memory", max_spill_bytes=None, max_local_bytes=None)

    detailed = analyze_prim_func(
        func, dict(settings, memory_diagnostics=True), target=TARGET, device_limits=LIMITS
    )

    def fail(*args, **kwargs):
        pytest.fail("lean memory mode must not invoke score-independent detailed analysis")

    monkeypatch.setattr("tilelang.tiletune.engine._propagate_tiles", fail)
    monkeypatch.setattr("tilelang.tiletune.engine.register_pressure.analyze_register_pressure", fail)
    monkeypatch.setattr("tilelang.tiletune.engine.analyze_live_tiles", fail)
    monkeypatch.setattr("tilelang.tiletune.engine.shared_memory.analyze_shared_memory", fail)
    lean = analyze_prim_func(func, settings, target=TARGET, device_limits=LIMITS)

    assert lean["tile_cost"]["score"] == detailed["tile_cost"]["score"]
    assert lean["tile_cost"]["tie_break_score"] == detailed["tile_cost"]["tie_break_score"]
    assert lean["tile_propagation"]["precision"] == "disabled"
    assert lean["pressure"]["tile_liveness"]["precision"] == "disabled"
    assert lean["pressure"]["tile_liveness"]["computing_threads_estimate"] == 128
    assert lean["modules"]["memory_traffic"]["dependencies"] is None
    assert lean["modules"]["shared_memory"]["precision"] == "disabled"


def test_lean_memory_mode_retains_hard_launch_limit():
    result = analyze_prim_func(
        gemm(threads=2048),
        dict(ranking_metric="memory", max_spill_bytes=None, max_local_bytes=None),
        target=TARGET,
        device_limits=LIMITS,
    )
    assert result["pressure"]["tile_liveness"]["precision"] == "disabled"
    assert result["pressure"]["decision"]["would_reject"]
    assert result["pressure"]["decision"]["classification"] == "resource_violation"


@pytest.mark.parametrize("target", [TARGET, {"kind": "hip", "mcpu": "gfx950"}])
def test_memory_model_uses_backend_inputs_without_kernel_family_rules(target):
    @T.prim_func
    def copy(A: T.Tensor((125, 8), "float32"), B: T.Tensor((125, 8), "float32")):
        with T.Kernel(125, threads=128) as block:
            tile = T.alloc_fragment((8,), "float32")
            T.copy(A[block, :], tile)
            T.copy(tile, B[block, :])

    for units, expected in ((132, 64), (64, 128)):
        result = analyze_prim_func(copy, dict(ranking_metric="memory"), target=target, device_limits={"sm_count": units})
        assert result["tile_cost"]["logical_byte_waves"] == expected
        assert result["specialization"]["name"] == "generic"
        assert result["pressure"]["target_model"]["kind"] == target["kind"]


def test_explicit_resource_policy_still_rejects():
    result = analyze_prim_func(gemm(), dict(ranking_metric="memory", register_cap=1), target=TARGET, device_limits=LIMITS)
    records = [dict(index=0, tile_cost=result["tile_cost"], pre_lowering=result["pressure"]["decision"])]
    assert result["pressure"]["decision"]["would_reject"]
    assert result["tile_cost"]["score"] > 0
    assert select_top_k(rank_records(records), 1) == []


def test_tcgen05_strict_register_policy_does_not_propagate_tensor_memory(monkeypatch):
    @T.prim_func
    def kernel(A: T.Tensor((128, 32), "float16"), B: T.Tensor((32, 128), "float16"), C: T.Tensor((128, 128), "float32")):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((128, 32), "float16")
            b = T.alloc_shared((32, 128), "float16")
            c = T.alloc_tmem((128, 128), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.tcgen05_gemm(a, b, c, clear_accum=True, mbar=None)
            T.copy(c, C)

    def fail(*args, **kwargs):
        pytest.fail("tensor-memory accumulators do not require register-demand propagation")

    monkeypatch.setattr("tilelang.tiletune.engine._propagate_tiles", fail)
    result = analyze_prim_func(
        kernel,
        dict(ranking_metric="memory", register_cap=1),
        target={"kind": "cuda", "arch": "sm_100a"},
        device_limits={**LIMITS, "sm_count": 148},
    )
    assert result["tile_cost"]["score"] is not None
    assert result["pressure"]["modeled_lower_bound"] is None
    assert not result["pressure"]["decision"]["would_reject"]


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


def test_bound_aware_key_remains_u_eff_depth_events():
    access = [dict(operation=0, bytes=64, visits=4)]
    neutral = score_memory(access, 132, 132, pipeline_depth=3, launch_underfill=True)
    penalized = score_memory(
        access,
        132,
        132,
        pipeline_depth=3,
        occupancy_penalty=4,
        launch_underfill=True,
    )
    u_eff = penalized["adjusted_logical_byte_waves"]
    expected = 65535 * u_eff * (u_eff + 1) // 2
    expected += (65535 - penalized["pipeline_depth"]) * (u_eff + 1)
    expected += penalized["logical_memory_access_waves"]
    assert penalized["score"] == expected
    assert penalized["launch_waves_component"] is False
    assert penalized["launch_target_waves"] == 3
    assert penalized["launch_underfill_shortfall_blocks"] == 264
    assert neutral["adjusted_logical_byte_waves"] == 753
    assert penalized["tie_break_score"] == neutral["tie_break_score"]
    assert u_eff == 4 * neutral["adjusted_logical_byte_waves"]
    with pytest.raises(ValueError, match="occupancy_penalty"):
        score_memory(access, 132, 132, occupancy_penalty=0)


def test_bound_aware_launch_underfill_ends_at_three_sm_waves():
    access = [dict(operation=0, bytes=64, visits=4)]
    underfilled = score_memory(access, 395, 132, launch_underfill=True)
    full = score_memory(access, 396, 132, launch_underfill=True)
    assert underfilled["launch_underfill_shortfall_blocks"] == 1
    assert underfilled["adjusted_logical_byte_waves"] > underfilled["logical_byte_waves"]
    assert full["launch_underfill_shortfall_blocks"] == 0
    assert full["adjusted_logical_byte_waves"] == full["logical_byte_waves"]


def test_bound_aware_metric_stays_lightweight(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("bound-aware memory mode invoked a timing, occupancy, or specialization model")

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


def test_bound_aware_reports_and_applies_the_coarse_gate(tmp_path):
    @T.prim_func
    def repeated_gemm(
        A: T.Tensor((32, 32), "float16"),
        B: T.Tensor((32, 32), "float16"),
        C: T.Tensor((32, 32), "float32"),
    ):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((32, 32), "float16")
            b = T.alloc_shared((32, 32), "float16")
            c = T.alloc_fragment((32, 32), "float32")
            T.clear(c)
            for _ in T.Pipelined(256, num_stages=2):
                T.copy(A, a)
                T.copy(B, b)
                T.gemm(a, b, c)
            T.copy(c, C)

    path = tmp_path / "bound-aware.json"
    limits = {**LIMITS, "max_blocks_per_sm": 1}
    result = analyze_prim_func(
        repeated_gemm,
        {"ranking_metric": "bound_aware", "facts_path": str(path)},
        target={"kind": "cuda", "arch": "sm_100a"},
        device_limits=limits,
    )
    bound = result["modules"]["bound"]
    memory = result["modules"]["memory_traffic"]
    assert bound["bound"] == "compute"
    assert bound["ridge_flops_per_byte"] == 281.25
    assert bound["occupancy_penalty"] == 2
    assert result["tile_cost"]["ranking_metric"] == "bound_aware"
    assert result["modules"]["ranking"]["launch_target_waves"] == 3
    assert (
        result["modules"]["ranking"]["launch_underfill_shortfall_blocks"]
        == 3 * limits["sm_count"] - 1
    )
    expected = score_memory(
        memory["accesses"],
        memory["grid_blocks"],
        limits["sm_count"],
        memory["pipeline_depth"],
        occupancy_penalty=bound["occupancy_penalty"],
        launch_underfill=True,
    )
    assert result["tile_cost"]["score"] == expected["score"]
    assert json.loads(path.read_text())["backend"] == "bound_aware.v1"


def test_bound_aware_keeps_a_pure_copy_on_memory_order():
    @T.prim_func
    def kernel(A: T.Tensor((4096,), "float16"), B: T.Tensor((4096,), "float16")):
        with T.Kernel(1, threads=128):
            tile = T.alloc_shared((96,), "float16")
            for k in T.Pipelined(T.ceildiv(4096, 96), num_stages=2):
                T.copy(A[k * 96], tile)
                T.copy(tile, B[k * 96])

    result = analyze_prim_func(
        kernel,
        {"ranking_metric": "bound_aware"},
        target={"kind": "cuda", "arch": "sm_80"},
        device_limits=LIMITS,
    )
    assert result["modules"]["bound"]["bound"] == "memory"
    assert result["modules"]["bound"]["occupancy_penalty"] == 1


def test_data_dependent_addresses_keep_resolved_memory_volume():
    @T.prim_func
    def indirect(
        A: T.Tensor((64,), "float32"), offsets: T.Tensor((1,), "int32"), B: T.Tensor((32,), "float32")
    ):
        with T.Kernel(1, threads=32):
            offset = T.bind(offsets[0])
            for i in T.Parallel(32):
                B[i] = A[offset + i]

    result = analyze_prim_func(indirect, dict(ranking_metric="memory"), target=TARGET, device_limits=LIMITS)
    assert result["tile_cost"]["score"] is not None
    assert not result["tile_cost"]["unknown"]


def test_bounded_while_is_read_from_primfunc():
    @T.prim_func
    def persistent_copy(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        with T.Kernel(4, threads=32) as block:
            ring = T.alloc_shared((3, 4), "float32")
            state = T.alloc_local((1,), "int32")
            state[0] = block
            with T.While(state[0] < 8):
                for k in T.serial(2):
                    phase = state[0] * 2 + k
                    T.copy(A[state[0] * 4 : state[0] * 4 + 4], ring[phase % 3, :])
                    T.copy(ring[phase % 3, :], B[state[0] * 4 : state[0] * 4 + 4])
                state[0] = state[0] + 4

    result = analyze_prim_func(persistent_copy, dict(ranking_metric="memory"), target=TARGET, device_limits=LIMITS)
    assert result["tile_cost"]["score"] is not None
    assert not result["tile_cost"]["unknown"]
    assert result["tile_cost"]["pipeline_depth"] == 1
    assert [(access["visits"], access["visit_precision"]) for access in result["tile_cost"]["accesses"]] == [
        (4, "conservative"),
        (4, "conservative"),
    ]


def test_conditionally_advanced_while_remains_unknown():
    @T.prim_func
    def conditional(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        with T.Kernel(4, threads=32) as block:
            state = T.alloc_local((1,), "int32")
            state[0] = block
            with T.While(state[0] < 8):
                T.copy(A[state[0] * 4 : state[0] * 4 + 4], B[state[0] * 4 : state[0] * 4 + 4])
                if block == 0:
                    state[0] = state[0] + 4

    result = analyze_prim_func(conditional, dict(ranking_metric="memory"), target=TARGET, device_limits=LIMITS)
    assert result["tile_cost"]["score"] is None
    assert "unmodeled scope: While" in result["tile_cost"]["unknown"]


def test_fp8_persistent_scheduler_and_manual_pipeline_are_read_from_primfunc(monkeypatch):
    from tilelang.carver.arch import driver
    from experiments.gemm_fp8.cases import cases
    from experiments.gemm_fp8.kernel import make_case
    from experiments.gemm_fp8.spaces import get_configs

    monkeypatch.setattr(driver, "get_num_sms", lambda: 148)
    func = make_case(cases(holdout=True)[1]).build(**get_configs()[419])
    result = analyze_prim_func(
        func,
        dict(ranking_metric="memory"),
        target={"kind": "cuda", "arch": "sm_100a"},
        device_limits={**LIMITS, "sm_count": 148},
    )
    assert result["tile_cost"]["score"] is not None
    assert not result["tile_cost"]["unknown"]
    assert result["tile_cost"]["pipeline_depth"] == 6
    assert all(
        access["visit_precision"] == "conservative" for access in result["tile_cost"]["accesses"]
    )


def test_opaque_global_effect_inside_a_store_remains_unknown():
    @T.prim_func
    def opaque(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(32):
                B[i] = T.call_extern("float32", "opaque_memory_effect", A.data)

    result = analyze_prim_func(opaque, dict(ranking_metric="memory"), target=TARGET, device_limits=LIMITS)
    assert result["tile_cost"]["score"] is None
    assert result["tile_cost"]["unknown"]


def test_wave_rounding_memory_ties_and_measurement_independence():
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
    assert [row["index"] for row in expected] == [4, 7, 2]
    assert [row["rank"] for row in expected] == [2, 2, 3]
    assert [row["position"] for row in expected] == [1, 2, 3]
    assert select_top_k(expected, 1) == [4, 7]
    assert select_top_k(expected, 1, strict_budget=True) == []
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


def test_exact_score_orders_depth_before_requests_without_float_rounding():
    size = 2**30
    fewer = [dict(operation=0, bytes=size, visits=1)]
    more = [dict(operation=0, bytes=1, visits=size)]
    scores = [
        score_memory(fewer, 1, 1, 65535)["score"],
        score_memory(more, 1, 1, 65535)["score"],
        score_memory(fewer, 1, 1, 1)["score"],
        score_memory(more, 1, 1, 1)["score"],
        score_memory([dict(operation=0, bytes=size + 1, visits=1)], 1, 1, 65535)["score"],
    ]
    assert all(type(value) is int and value > 2**53 for value in scores)
    assert scores == sorted(set(scores))
