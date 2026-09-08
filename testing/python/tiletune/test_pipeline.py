"""Pipeline coverage and compute ownership, independent of measured timings."""

import pytest
import tilelang.language as T
from tilelang.tiletune import analyze_prim_func
from tilelang.tiletune.pipeline import estimate_pipeline_cycles
from test_cost import LIMITS
from test_modules import PROFILE, TARGET


def matrix_pipeline(threads=128, stages=3):
    @T.prim_func
    def main(A: T.Tensor((64, 128), "float16"), B: T.Tensor((128, 64), "float16"), C: T.Tensor((64, 64), "float32")):
        with T.Kernel(8, threads=threads):
            a = T.alloc_shared((64, 32), "float16")
            b = T.alloc_shared((32, 64), "float16")
            c = T.alloc_fragment((64, 64), "float32")
            T.clear(c)
            for k in T.Pipelined(4, num_stages=stages):
                T.copy(A[:, k * 32 : k * 32 + 32], a)
                T.copy(B[k * 32 : k * 32 + 32, :], b)
                T.gemm(a, b, c)
            T.copy(c, C)

    return main


def pipeline(threads=128, stages=3, profile=None, **kwargs):
    return analyze_prim_func(
        matrix_pipeline(threads, stages), {"performance_model": profile or PROFILE}, target=TARGET, device_limits=LIMITS, **kwargs
    )["modules"]["pipeline_overlap"]


def test_stage_zero_serial_loop_is_scored_without_overlap():
    result = pipeline(stages=0)
    assert not result["unknown"]
    assert result["iterations"]["max"] == 4
    assert result["num_stages"] == 0
    assert result["effective_buffer_depth"] == 1
    assert not result["overlap_eligible"]
    assert result["producer_copies_per_iteration"] == 2
    timing = result["timing"]
    assert timing["steady_state_interval_cycles"] == timing["input_ready_latency_cycles"] + timing["consumer_cycles_per_iteration"]


def test_wgmma_participants_use_consumers_and_honor_pass_overrides():
    for threads in (128, 256):
        func = matrix_pipeline(threads).with_attr("tilelang_pass_configs", {"tl.disable_wgmma": True})
        before = func.script()
        for override, instruction in ((None, "cuda.mma"), ({"tl.disable_wgmma": False}, "cuda.wgmma")):
            result = analyze_prim_func(func, target=TARGET, device_limits=LIMITS, pass_configs=override)
            phase = next(p for p in result["modules"]["pipeline_overlap"]["phases"] if p["work"]["gemm_flops"])
            participants = phase["compute_participants"]
            assert participants["instruction"] == instruction
            assert participants["consumer_threads"] == threads
            assert participants["warpgroups"] == (threads // 128 if instruction == "cuda.wgmma" else None)
        assert func.script() == before


def test_per_warpgroup_limit_and_shared_sm_limit_both_apply():
    profile = dict(
        PROFILE,
        global_bytes_per_cycle=8192,
        shared_bytes_per_cycle=8192,
        gemm_flops_per_cycle=32768,
        elementwise_ops_per_cycle=4096,
        copy_latency_cycles=0,
        barrier_cycles=0,
        wgmma_flops_per_cycle_per_warpgroup=512,
    )
    one, two = [pipeline(t, profile=profile) for t in (128, 256)]
    assert one["timing"]["consumer_cycles_per_iteration"] == 2 * two["timing"]["consumer_cycles_per_iteration"]
    # Additional CTAs share the SM ceiling but do not reduce a single CTA's
    # dependency on the compute rate of its own consumer group(s).
    assert estimate_pipeline_cycles(one, 2)["consumer_cycles_per_iteration"] == one["timing"]["consumer_cycles_per_iteration"]
    assert estimate_pipeline_cycles(one, 128)["consumer_cycles_per_iteration"] > one["timing"]["consumer_cycles_per_iteration"]


def test_unknown_compute_participants_do_not_invent_group_throughput():
    result = pipeline()
    result["performance_model"] = dict(PROFILE, wgmma_flops_per_cycle_per_warpgroup=512)
    phase = next(p for p in result["phases"] if p["work"]["gemm_flops"])
    phase["compute_participants"] = {"precision": "unknown"}
    assert estimate_pipeline_cycles(result) is None


@pytest.mark.parametrize("stages", [0, 1, 2, 3])
def test_profile_only_changes_ranking_not_pressure_or_traffic(stages):
    first = analyze_prim_func(matrix_pipeline(stages=stages), target=TARGET, device_limits=LIMITS)
    second = analyze_prim_func(
        matrix_pipeline(stages=stages),
        {"ranking_metric": "pipeline_time", "performance_model": dict(PROFILE, wgmma_flops_per_cycle_per_warpgroup=512)},
        target=TARGET,
        device_limits=LIMITS,
    )
    assert first["pressure"]["decision"] == second["pressure"]["decision"]
    assert first["pressure"]["modeled_lower_bound"] == second["pressure"]["modeled_lower_bound"]
    assert first["tile_cost"]["traffic_bytes_per_block"] == second["tile_cost"]["traffic_bytes_per_block"]
    assert second["tile_cost"]["score"] is not None
