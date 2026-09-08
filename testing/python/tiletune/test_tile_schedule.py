"""Buffer dependencies, fragment collectives and per-CTA work, without GPU timing."""

from types import SimpleNamespace
import random

import pytest
import tilelang.language as T
from tilelang import tvm
from tvm import tirx as tir
from tilelang.tiletune import analyze_prim_func
from tilelang.tiletune.cta_work import collect_cta_work, estimate_grid_cycles
from tilelang.tiletune.reduction import fragment_reduction_work
from tilelang.tiletune.tile_schedule import buffer_transition, repeat_transition
from test_cost import LIMITS
from test_modules import PROFILE, TARGET


def event_oracle(copies, consumers, depth, bandwidth, latency, barrier, iterations):
    # Deliberately expand a small schedule into concrete buffer events; the
    # production recurrence must agree without expanding the inner loop.
    releases = [[0.0] * depth for _ in copies]
    consumer = issue = service = 0.0
    for iteration in range(iterations):
        slot = iteration % depth
        ready = []
        for i, copy in enumerate(copies):
            issue = max(issue, releases[i][slot])
            service = max(service, issue) + copy["bytes"] / bandwidth
            ready.append(service + latency)
        for op, cycles in consumers:
            needs = [ready[i] for i, c in enumerate(copies) if c["first_consumer"] == op]
            consumer = max([consumer, *needs]) + cycles + len(needs) * barrier
            for i, copy in enumerate(copies):
                if copy["last_consumer"] == op:
                    releases[i][slot] = consumer
    return consumer


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_periodic_schedule_matches_buffer_event_oracle(depth):
    rng = random.Random(57)
    for _ in range(10):
        copies = [
            dict(bytes=rng.randrange(1, 512), first_consumer=1, last_consumer=1),
            dict(bytes=rng.randrange(1, 512), first_consumer=3, last_consumer=3),
        ]
        consumers = [(i, rng.randrange(1, 100)) for i in (1, 2, 3)]
        bandwidth, latency, barrier = 8, rng.randrange(1, 50), 3
        transition = buffer_transition(copies, consumers, depth, bandwidth, latency, barrier)
        for iterations in (0, 1, 2, 11, 25):
            expected = event_oracle(copies, consumers, depth, bandwidth, latency, barrier, iterations)
            assert repeat_transition(transition, iterations)[0] == pytest.approx(expected)


def test_independent_release_allows_overlap_with_one_slot():
    copies = [dict(bytes=100, first_consumer=1, last_consumer=1), dict(bytes=100, first_consumer=3, last_consumer=3)]
    consumers = [(1, 10), (2, 100), (3, 10)]
    separate = buffer_transition(copies, consumers, 1, 10, 40, 0)
    locked = buffer_transition([dict(c, last_consumer=3) for c in copies], consumers, 1, 10, 40, 0)
    assert repeat_transition(separate, 16)[0] < repeat_transition(locked, 16)[0]
    # Huge loop extents do not allocate one event per iteration.
    huge = repeat_transition(separate, 10**9)[0]
    assert huge >= 120 * 10**9
    with pytest.raises(ValueError, match="nonnegative"):
        repeat_transition(separate, -1)


@pytest.mark.parametrize("replication", [1, 2])
def test_fragment_local_work_and_lane_collectives(replication):
    layout = T.Fragment((8, 64), forward_fn=lambda i, j: (i * 4 + j % 4, j // 4))
    if replication > 1:
        layout = layout.replicate(replication)
    result = fragment_reduction_work(layout, [8, 64], 1)
    assert result["local_pairs"] == 8 * 4 * 15 * replication
    assert result["shuffle_pairs"] == 8 * 4 * 2 * replication
    assert result["lane_widths"] == [4]
    assert result["values_per_lane"] == [16]


def test_interwarp_reduction_is_unknown():
    layout = T.Fragment((1, 64), forward_fn=lambda i, j: (j, i))
    with pytest.raises(ValueError, match="inter-warp"):
        fragment_reduction_work(layout, [1, 64], 1)


def real_attention(causal=False, stages=1, bn=256):
    from examples.flash_attention.example_mha_tiletune import make_attention

    func = make_attention(causal=causal)(128, bn, stages, 256)
    before = func.script()
    result = analyze_prim_func(
        func,
        dict(ranking_metric="pipeline_time", performance_model=PROFILE, attention_spill_budget_registers_per_thread=32),
        target=TARGET,
        device_limits=LIMITS,
    )
    assert func.script() == before
    return result


def test_attention_dependencies_and_physical_reductions():
    result = real_attention()
    pipe = result["modules"]["pipeline_overlap"]
    phases = {p["operation"]: p for p in pipe["phases"]}
    copies = pipe["producer_buffers"]
    assert len(copies) == 2
    assert [phases[c["last_consumer"]]["phase"] for c in copies] == ["qk_gemm", "pv_gemm"]
    assert all(c["bytes"] == 128 * 256 * 2 for c in copies)
    reductions = [p["reduction"] for p in pipe["phases"] if p["reduction"]]
    assert {r["operator"] for r in reductions} == {"sum", "max"}
    assert all(r["dtype"] == "float32" and r["local_pairs"] == 32256 and r["shuffle_pairs"] == 1024 for r in reductions)
    serial = real_attention(stages=0)["modules"]["pipeline_overlap"]
    assert pipe["timing"]["cycles"] < serial["timing"]["cycles"]
    assert pipe["timing"]["schedule_model"] == "per-buffer max-plus recurrence"


def test_default_reference_has_a_pipeline_score():
    from examples.flash_attention.example_mha_tiletune import REFERENCE_CONFIG, get_configs

    assert get_configs().index(REFERENCE_CONFIG) == 85
    result = real_attention(stages=REFERENCE_CONFIG["num_stages"], bn=REFERENCE_CONFIG["block_N"])
    assert result["modules"]["ranking"]["score"] is not None


@pytest.mark.parametrize("bn,mean,maximum", [(128, 16.5, 32), (256, 8.5, 16)])
def test_actual_causal_cta_counts(bn, mean, maximum):
    result = real_attention(causal=True, bn=bn)
    pipe = result["modules"]["pipeline_overlap"]
    work = pipe["cta_work"]
    assert work["precision"] == "exact"
    assert work["mean_iterations"] == mean
    assert work["max_iterations"] == maximum
    assert work["grid_blocks"] == 512
    assert work["repetitions"] == 16
    assert sum(g["count"] for g in work["groups"]) == 32
    timing = estimate_grid_cycles(work, lambda n: {"cycles": n * 100}, 132, 4)
    assert timing["cycles"] < maximum * 100 * 4
    assert timing["cycles"] >= timing["work_lower_bound_cycles"]


def test_cta_counts_use_ir_variable_and_preserve_partial_tail():
    block = tir.Var("renamed_query_axis", "int32")
    other = tir.Var("other", "int32")
    col = SimpleNamespace(block_domains={"blockIdx.x": (other, tvm.ir.Range(0, 3)), "blockIdx.y": (block, tvm.ir.Range(0, 5))})
    loop = SimpleNamespace(extent=tir.min(3, tir.floordiv((block + 1) * 3 + 3, 4)))
    result = collect_cta_work(col, loop)
    assert result["groups"] == [dict(iterations=1, count=3), dict(iterations=2, count=3), dict(iterations=3, count=9)]
    assert result["mean_iterations"] == 2.4
    assert result["grid_blocks"] == 15
    assert collect_cta_work(col, loop, max_axis_points=4)["precision"] == "unknown"
    loop.extent = block + other
    assert collect_cta_work(col, loop)["precision"] == "unknown"


def test_uniform_grid_keeps_wave_formula_and_missing_timing_stays_unknown():
    work = dict(precision="exact", groups=[dict(iterations=10**9, count=512)], repetitions=1, grid_blocks=512)
    timing = estimate_grid_cycles(work, lambda n: {"cycles": n * 100}, 132, 4)
    assert timing["cycles"] == 4 * 100 * 10**9
    assert timing["method"] == "uniform CTA waves"
    assert estimate_grid_cycles(work, lambda n: None, 132, 4) is None
