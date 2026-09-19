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
    assert timing["input_ready_latency_cycles"] == timing["copy_service_cycles_per_iteration"]
    phase_cycles = sum(
        item["cycles"]
        for item, phase in zip(timing["phase_cycles"], result["phases"])
        if phase["inside_loop"]
    )
    assert timing["consumer_cycles_per_iteration"] == phase_cycles + 2 * PROFILE["barrier_cycles"]


def test_stage_zero_direct_copy_ranking_does_not_use_async_readiness_latency():
    first = pipeline(stages=0)
    second = pipeline(stages=0, profile=dict(PROFILE, copy_latency_cycles=800))
    without_latency = dict(PROFILE)
    without_latency.pop("copy_latency_cycles")
    assert first["timing"] == second["timing"]
    assert pipeline(stages=0, profile=without_latency)["timing"] == first["timing"]


def test_dense_gemm_uses_profiled_l2_rate_only_for_a_resident_k_slice():
    resident = pipeline(profile=dict(PROFILE, cached_global_bytes_per_cycle=128, l2_cache_bytes=8192))
    cache = resident["inter_cta_cache"]
    assert cache["status"] == "applied"
    assert cache["slice_working_set_bytes"] == 8192
    assert cache["logical_input_bytes_per_iteration"] == 8192
    assert cache["dram_unique_bytes_per_cta_iteration"] == 1024
    assert cache["dram_equivalent_service_bytes_per_iteration"] == 4608
    assert resident["timing"]["copy_service_cycles_per_iteration"] == 4608 / PROFILE["global_bytes_per_cycle"]
    streaming = pipeline(profile=dict(PROFILE, cached_global_bytes_per_cycle=128, l2_cache_bytes=8191))
    assert streaming["inter_cta_cache"] is None
    assert streaming["timing"]["copy_service_cycles_per_iteration"] == 8192 / PROFILE["global_bytes_per_cycle"]


def test_blackwell_tma_pipeline_uses_native_warp_specialization_policy():
    result = analyze_prim_func(
        matrix_pipeline(threads=256, stages=3),
        {"performance_model": PROFILE},
        target={"kind": "cuda", "arch": "sm_100a"},
        device_limits=LIMITS,
    )
    policy = result["modules"]["warp_specialization"]
    assert policy["status"] == "predicted"
    assert policy["producer_threads"] == 128
    assert policy["consumer_threads"] == 256
    assert policy["launch_threads"] == 384
    assert policy["register_reservation_per_block"] == 128 * 24 + 256 * 240
    assert result["modules"]["waves"]["launch_threads"] == 384
    assert result["modules"]["pipeline_overlap"]["overlap_eligible"]
    assert result["tile_cost"]["score"] is not None


def test_blackwell_serial_mma_occupancy_includes_native_operand_fragments():
    result = analyze_prim_func(
        matrix_pipeline(threads=256, stages=0),
        {"performance_model": PROFILE},
        target={"kind": "cuda", "arch": "sm_100a"},
        device_limits=LIMITS,
    )
    waves = result["modules"]["waves"]
    assert result["pressure"]["mma_operand_registers"]
    assert waves["registers_per_block_estimate"] > waves["logical_tile_registers_per_block_estimate"]
    assert "native MMA operand fragments" in waves["register_estimate_basis"]


def test_grouped_gemm_outer_dispatch_does_not_hide_inner_blackwell_pipeline():
    from experiments.grouped_gemm.cases import cases
    from experiments.grouped_gemm.kernel import make_case
    from experiments.grouped_gemm.spaces import get_configs

    workload = cases(holdout=True)[0]
    case = make_case(workload)
    configs = get_configs()
    results = []
    for candidate in configs:
        func = case.build(**candidate)
        results.append(
            analyze_prim_func(
                func,
                {"performance_model": dict(PROFILE, tcgen05_gemm_flops_per_cycle=4096)},
                target={"kind": "cuda", "arch": "sm_100a"},
                device_limits=LIMITS,
            )
        )
    n = workload.parameters["n"]
    assert [r["modules"]["waves"]["grid_blocks"] for r in results] == [
        (n + candidate["block_N"] - 1) // candidate["block_N"] for candidate in configs
    ]
    assert [r["modules"]["waves"]["launch_threads"] for r in results] == [
        candidate["threads"] + (128 if candidate["num_stages"] else 0) for candidate in configs
    ]
    for candidate, result in zip(configs, results):
        model = result["modules"]["pipeline_overlap"].get("timing", {}).get("schedule_model")
        if candidate["num_stages"] == 0:
            assert model == "grouped GEMM serial copy/consumer loop"
        elif result["tile_cost"]["score"] is not None:
            assert model == "grouped GEMM per-buffer max-plus recurrence"
    rejected = [result for result in results if result["tile_cost"]["score"] is None]
    assert len(rejected) == 2
    assert all("estimated block resources exceed device limits" in result["tile_cost"]["unknown"] for result in rejected)


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
