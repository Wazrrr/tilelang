"""Ampere compiler plans and pipeline scoring, independent of candidate timing."""

import pytest
import tilelang.language as T
from tilelang.tiletune import analyze_prim_func
from tilelang.tiletune.ampere import schedule_cycles
from tilelang.tiletune.compute import operation_work
from tilelang.tiletune.src.collector import _Collector
from test_portability import AMPERE, LIMITS, ampere_profile
from test_pipeline import matrix_pipeline
from test_modules import attention


def profile():
    return dict(ampere_profile(), rsqrt_ops_per_cycle=16, async_copy_issue_bytes_per_cycle=128, async_copy_latency_cycles=200)


def analyze(func, pass_configs=None):
    return analyze_prim_func(
        func,
        {"ranking_metric": "pipeline_time", "performance_model": profile()},
        target=AMPERE,
        device_limits=LIMITS,
        pass_configs=pass_configs,
    )


@pytest.mark.parametrize("stages", [0, 1, 2, 3])
@pytest.mark.parametrize("factory", [matrix_pipeline, attention])
def test_compiler_plans_score_without_changing_ir(factory, stages):
    func = factory(stages=stages)
    before = func.script()
    result = analyze(func)
    pipeline = result["modules"]["pipeline_overlap"]
    assert func.script() == before
    assert not pipeline["unknown"]
    assert result["tile_cost"]["score"] > 0
    assert pipeline["overlap_eligible"] == bool(stages)
    if stages:
        assert pipeline["ampere_schedule"]["status"] == "predicted"
        assert "compiler-ordered" in pipeline["timing"]["schedule_model"]
    assert result["pressure"]["warp_specialization"]["status"] == "not_applicable"
    assert all("unknown" not in entry for entry in result["pressure"]["ampere_mma_operand_registers"].values())


def test_parallel_scalar_work_counts_tile_elements():
    @T.prim_func
    def main(A: T.Tensor((64, 64), "float32"), B: T.Tensor((64, 64), "float32")):
        with T.Kernel(1, threads=128):
            for i, j in T.Parallel(64, 64):
                B[i, j] = T.exp(A[i, j]) + T.rsqrt(A[i, j])

    work = operation_work(_Collector(main).operations[0])
    assert work["exp_ops"] == work["rsqrt_ops"] == 4096
    assert work["elementwise_ops"] == 4096


@pytest.mark.parametrize("case_index", [0, 1])
@pytest.mark.parametrize("missing", [None, "log_ops_per_cycle", "reduction_local_max_float16_per_cycle"])
def test_online_softmax_requires_measured_log_and_half_max(case_index, missing):
    from examples.online_softmax.online_softmax import softmax_kernel

    shape = ((1536, 4096), (1031, 1537))[case_index]
    func = softmax_kernel.get_tir(T.Tensor(shape, "float16"), BLOCK_M=4, BLOCK_N=256, threads=128, dtype="float16")
    before = func.script()
    rates = dict(
        profile(),
        log_ops_per_cycle=16,
        reduction_local_max_float16_per_cycle=128,
        reduction_shuffle_max_float16_per_cycle=32,
    )
    from tiletune_core.profile_schema import CONSUMER_RATE_FIELDS

    rates["consumer_rates"] = {"128": {key: value for key, value in rates.items() if key in CONSUMER_RATE_FIELDS}}
    if missing:
        rates.pop(missing)
    result = analyze_prim_func(func, dict(ranking_metric="pipeline_time", performance_model=rates), target=AMPERE, device_limits=LIMITS)
    assert func.script() == before
    if missing:
        assert result["tile_cost"]["score"] is None
    else:
        assert result["tile_cost"]["score"] > 0
        pipeline = result["modules"]["pipeline_overlap"]
        assert not pipeline["unknown"]
        assert sum(p["work"].get("log_ops", 0) for p in pipeline["phases"]) > 0
        assert {p["reduction"]["dtype"] for p in pipeline["phases"] if p["reduction"]} == {"float16", "float32"}


@pytest.mark.parametrize("name", ["softmax", "rmsnorm", "reduce_sum", "kda_recurrent"])
def test_generic_reductions_use_compiler_ownership(name):
    if name == "kda_recurrent":
        from regression_kernels import recurrent_program

        func = recurrent_program(1, 8, 256, 64, 64, "float16", 16, 128)
        pass_configs = {}
    elif name == "softmax":
        from regression_kernels import softmax_program

        # Preserve the FP32-reduction graph this profile was calibrated for.
        # The online example also needs log2 and FP16-max rates.
        func = softmax_program(1024, 2048, "float16", 4, 256, 128, 1, 1)
        pass_configs = {}
    else:
        from regression_kernels import row_reduction_program

        func = row_reduction_program(4096, 4096, "float16", name, 1e-6, 1, 128)
        pass_configs = {}
    before = func.script()
    result = analyze(func, pass_configs)
    pipeline = result["modules"]["pipeline_overlap"]
    assert not pipeline["unknown"]
    assert result["tile_cost"]["score"] > 0
    reductions = [p["reduction"] for p in pipeline["phases"] if p["reduction"]]
    assert reductions and all(r["precision"] == "predicted" for r in reductions)
    assert all(r["shared_pairs"] > 0 and r["barrier_rounds"] > 0 for r in reductions)
    assert func.script() == before
    if name == "kda_recurrent":
        inside = [p for p in pipeline["phases"] if p["inside_loop"]]
        assert any(p["external_work"]["read_bytes"] > 0 for p in inside)
        assert any(p["external_work"]["write_bytes"] > 0 for p in inside)
        assert all(
            r["local_outputs_per_thread"] == 4 and r["barrier_rounds"] == 16 and r["workspace_reuse_barriers"] == 4 for r in reductions
        )
        # Two distinct G values per physical thread, reused across four local
        # value columns. Offline cubin inspection confirms two EX2 sites.
        assert sum(p["work"]["exp_ops"] for p in inside) == 2 * 128
        # Delta has 16 logical outputs replicated over 32 reducing lanes.
        delta = next(p for p in inside if p["operation"] == 4)
        assert delta["work"]["elementwise_ops"] == 16 * 32 * 4
    if name == "rmsnorm":
        # The row-dependent rsqrt is hoisted out of the 32-value local loop.
        assert sum(p["work"].get("rsqrt_ops", 0) for p in pipeline["phases"]) == 128


@pytest.mark.parametrize("depth,expected", [(1, 99), (2, 66)])
def test_pipeline_startup_drain_and_buffer_reuse(depth, expected):
    # One async copy: issue=1, byte service=4, load-to-use=20 cycles.
    # One consumer: wait barrier=2, compute=10. Depth one serializes
    # copy/consumer pairs; depth two issues the next tile before consumption.
    copy = dict(operation=0, bytes=64, first_consumer=1, last_consumer=1)
    events = [dict(operations=[0], stage=0, async_group=0), dict(operations=[1], stage=1, async_group=-1)]
    if depth == 1:
        events.reverse()
    plan = dict(events=events, max_stage=1, buffer_depth=depth)
    rates = dict(
        global_bytes_per_cycle=16,
        async_copy_issue_bytes_per_cycle=64,
        async_copy_latency_cycles=20,
        copy_latency_cycles=0,
        barrier_cycles=2,
    )
    assert schedule_cycles(plan, [copy], {0: 0, 1: 10}, 0, rates, 1) == 0
    assert schedule_cycles(plan, [copy], {0: 0, 1: 10}, 1, rates, 1) == 33
    assert schedule_cycles(plan, [copy], {0: 0, 1: 10}, 3, rates, 1) == expected
    assert schedule_cycles(plan, [copy], {0: 0, 1: 10}, 100000000, rates, 1) > 1e9


@pytest.mark.parametrize("stages", [1, 2, 3])
@pytest.mark.parametrize("factory", [matrix_pipeline, attention])
def test_max_plus_matches_explicit_event_replay(factory, stages):
    from tilelang.tiletune.compute import estimate_phase_cycles

    pipeline = analyze(factory(stages=stages))["modules"]["pipeline_overlap"]
    plan, rates = pipeline["ampere_schedule"], profile()
    copies = {c["operation"]: c for c in pipeline["producer_buffers"]}
    events = plan["events"]
    owner = {op: e for e in events for op in e["operations"]}
    costs = {p["operation"]: estimate_phase_cycles(p, rates, 1) for p in pipeline["phases"]}
    for iterations in (1, 2, 3, 9):
        warp = service = 0.0
        ready, released = {}, {}
        for step in range(iterations + plan["max_stage"]):
            waited = set()
            for event in events:
                iteration = step - event["stage"]
                if not 0 <= iteration < iterations:
                    continue
                for op in event["operations"]:
                    if op in copies:
                        previous = iteration - plan["buffer_depth"]
                        if previous >= 0:
                            assert (op, previous) in released
                        warp += copies[op]["bytes"] / rates["async_copy_issue_bytes_per_cycle"]
                        service = max(warp, service) + copies[op]["bytes"] / rates["global_bytes_per_cycle"]
                        ready[op, iteration] = max(service, warp + rates["async_copy_latency_cycles"])
                    else:
                        groups = {owner[c]["async_group"] for c in copies if copies[c]["first_consumer"] == op}
                        for group in groups - waited:
                            members = [c for c in copies if owner[c]["async_group"] == group]
                            warp = max(warp, *(ready[c, iteration] for c in members)) + rates["barrier_cycles"]
                            waited.add(group)
                        warp += costs[op]
                        for c in copies:
                            if copies[c]["last_consumer"] == op:
                                released[c, iteration] = warp
        assert schedule_cycles(plan, list(copies.values()), costs, iterations, rates, 1) == pytest.approx(warp)


@pytest.mark.parametrize("kind", ["transpose", "atomic", "scan"])
def test_unsupported_operations_cannot_receive_zero_cost(kind):
    @T.prim_func
    def main(A: T.Tensor((32, 32), "float32"), B: T.Tensor((32, 32), "float32")):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((32, 32), "float32")
            b = T.alloc_shared((32, 32), "float32")
            c = T.alloc_fragment((32, 32), "float32")
            T.copy(A, a)
            if kind == "transpose":
                T.transpose(a, b)
                T.copy(b, c)
                T.copy(c, B)
            elif kind == "atomic":
                T.copy(a, c)
                T.atomic_add(B, c)
            else:
                T.cumsum(a, dim=1)
                T.copy(a, B)

    result = analyze(main)
    assert result["tile_cost"]["score"] is None
    assert "unresolved operation work" in result["modules"]["pipeline_overlap"]["unknown"]


@pytest.mark.parametrize("stage", [0, 1, 2, 3, 4])
def test_kda_example_pipeline_preserves_gate_work_and_value_tails(stage):
    from experiments.kda.cases import cases
    from experiments.kda.kernel import make_case

    case = make_case(cases(True)[0])
    func = case.build(block_DK=32, block_DV=48, num_stages=stage, threads=128)
    before = func.script()
    result = analyze(func)
    assert func.script() == before
    assert result["tile_cost"]["score"] > 0
    pipeline = result["modules"]["pipeline_overlap"]
    assert not pipeline["unknown"]
    assert sum(p["work"]["exp_ops"] for p in pipeline["phases"]) == 64 * 32
    assert pipeline["region_schedule"]["cta_work"]["grid_blocks"] == 128


def test_last_use_cast_retires_source_storage_without_changing_accumulator_bound():
    func = attention(stages=0, sequence=256, block_n=128)
    result = analyze(func)
    phases = result["pressure"]["tile_liveness"]["phases"]
    casts = [p for p in phases if p["streamed_cast_storage"]]
    assert len(casts) == 1
    phase = casts[0]
    logical = sum(b["logical_bits_with_modeled_replication"] for b in phase["buffers"])
    retired = phase["streamed_cast_storage"]["retired_bits_estimate"]
    assert retired > 0
    assert phase["packed_registers_per_block_estimate"] == (logical - retired + 31) // 32
    assert result["pressure"]["modeled_lower_bound"] == 64
