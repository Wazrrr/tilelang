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


@pytest.mark.parametrize("name", ["softmax", "rmsnorm", "reduce_sum", "kda_recurrent"])
def test_generic_reductions_use_compiler_ownership(name):
    from experiments.portable.spec import default_workloads, configurations, Device
    from experiments.portable.kernels import make_case

    workload = next(w for w in default_workloads() if w.name == name)
    case = make_case(workload)
    func = case.build(**configurations(workload, Device("ampere", AMPERE))[0])
    before = func.script()
    result = analyze(func, case.pass_configs)
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
