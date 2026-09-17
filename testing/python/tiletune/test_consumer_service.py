"""Independent consumer issue ceilings; no workload-specific timing factors."""

from copy import deepcopy
import pytest
from tilelang.tiletune import TileTuneConfig, analyze_prim_func
from tilelang.tiletune.pipeline import estimate_pipeline_cycles
from test_cost import LIMITS
from test_pipeline import matrix_pipeline
from test_modules import PROFILE, TARGET


def synthetic(threads=128):
    profile = dict(
        PROFILE,
        exp_ops_per_cycle=100,
        copy_latency_cycles=0,
        barrier_cycles=0,
        consumer_rates={"128": {"exp_ops_per_cycle": 20}, "256": {"exp_ops_per_cycle": 40}},
    )
    return dict(
        performance_model=profile,
        unknown=[],
        iterations={"max": 1},
        effective_buffer_depth=1,
        input_bytes_per_iteration=0,
        producer_copies_per_iteration=0,
        outside_loop_bytes=0,
        outside_loop_input_copies=0,
        overlap_eligible=False,
        phases=[
            dict(
                operation=0,
                inside_loop=True,
                consumer_threads=threads,
                work=dict(gemm_flops=0, shared_bytes=0, elementwise_ops=0, exp_ops=100, reduction_ops=0),
            )
        ],
    )


def test_consumer_and_aggregate_limits_are_separate():
    p = synthetic()
    before = deepcopy(p)
    assert estimate_pipeline_cycles(p)["cycles"] == 5
    assert estimate_pipeline_cycles(synthetic(256))["cycles"] == 2.5
    assert estimate_pipeline_cycles(p, concurrent_ctas=2)["cycles"] == 5
    assert estimate_pipeline_cycles(p, concurrent_ctas=8)["cycles"] == 8
    assert p == before


def test_unmeasured_consumer_domain_is_unknown_and_old_profile_is_compatible():
    p = synthetic(64)
    assert estimate_pipeline_cycles(p) is None
    p["performance_model"].pop("consumer_rates")
    assert estimate_pipeline_cycles(p)["cycles"] == 1


def test_local_and_shuffle_services_use_physical_work_separately():
    p = synthetic()
    p["performance_model"].update(reduction_local_sum_per_cycle=100, reduction_shuffle_sum_per_cycle=50)
    p["performance_model"]["consumer_rates"]["128"].update(reduction_local_sum_per_cycle=20, reduction_shuffle_sum_per_cycle=10)
    phase = p["phases"][0]
    phase["work"].update(exp_ops=0, reduction_ops=99)
    phase["reduction"] = dict(precision="predicted", dtype="float32", operator="sum", local_pairs=80, shuffle_pairs=20)
    assert estimate_pipeline_cycles(p)["cycles"] == 6
    assert estimate_pipeline_cycles(p, concurrent_ctas=8)["cycles"] == pytest.approx(9.6)


@pytest.mark.parametrize(
    "rows",
    [
        {},
        {"0": {"exp_ops_per_cycle": 10}},
        {"127": {"exp_ops_per_cycle": 10}},
        {"128": {"exp_ops_per_cycle": -1}},
        {"128": {"exp_ops_per_cycle": float("nan")}},
        {"128": {"exp_ops_per_cycle": True}},
        {"128": {"fitted_attention": 1}},
    ],
)
def test_invalid_consumer_profile(rows):
    with pytest.raises(ValueError):
        TileTuneConfig(performance_model={"consumer_rates": rows})


def test_producers_do_not_count_as_scalar_consumers():
    profile = dict(PROFILE, consumer_rates={"128": {"elementwise_ops_per_cycle": 50}})
    func = matrix_pipeline(128)
    before = func.script()
    result = analyze_prim_func(func, dict(ranking_metric="pipeline_time", performance_model=profile), target=TARGET, device_limits=LIMITS)
    pipe = result["modules"]["pipeline_overlap"]
    assert result["pressure"]["warp_specialization"]["launch_threads"] == 256
    assert {p["consumer_threads"] for p in pipe["phases"]} == {128}
    assert result["tile_cost"]["score"] is not None
    legacy = analyze_prim_func(func, target=TARGET, device_limits=LIMITS)
    assert result["pressure"] == legacy["pressure"]
    assert func.script() == before


def test_half_max_uses_its_own_rate_and_workspace_element_size():
    from tiletune_core.compute import estimate_phase_cycles

    phase = dict(
        consumer_threads=128,
        work=dict(gemm_flops=0, shared_bytes=0, elementwise_ops=0, exp_ops=0, reduction_ops=100),
        reduction=dict(
            precision="predicted",
            dtype="float16",
            operator="max",
            local_pairs=64,
            shuffle_pairs=32,
            shared_pairs=16,
            barrier_rounds=2,
        ),
    )
    rates = dict(
        shared_bytes_per_cycle=32,
        barrier_cycles=3,
        reduction_local_max_per_cycle=1000,
        reduction_shuffle_max_per_cycle=1000,
    )
    assert estimate_phase_cycles(phase, rates, 1) is None
    rates.update(reduction_local_max_float16_per_cycle=16, reduction_shuffle_max_float16_per_cycle=8)
    # Local + shuffle + shared combine + two-byte shared roundtrip + barriers.
    assert estimate_phase_cycles(phase, rates, 1) == 4 + 4 + 1 + 2 + 6


@pytest.mark.parametrize("kind", [0, 1])
def test_gpu_half_max_primitive(kind):
    import torch
    import tilelang
    from tilelang.tiletune.profiling.device_probes import half_max_primitive

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    threads = 128
    kernel = tilelang.compile(half_max_primitive(kind, 3, 1, threads), out_idx=[0], execution_backend="tvm_ffi")
    initial = (0.01 * (torch.arange(threads, device="cuda")[:, None] + torch.arange(8, device="cuda")[None, :] + 1)).half()
    expected = initial.clamp_min(0.5) if kind == 0 else torch.maximum(initial, initial[torch.arange(threads, device="cuda") ^ 1])
    torch.testing.assert_close(kernel()[0], expected.float().sum(1))


def test_gpu_log2_primitive():
    import torch
    import tilelang
    from tilelang.tiletune.profiling.device_probes import primitive

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    kernel = tilelang.compile(primitive(6, 3, 1), out_idx=[0], execution_backend="tvm_ffi")
    expected = 0.01 * (torch.arange(128, device="cuda")[:, None] + torch.arange(8, device="cuda")[None, :] + 1)
    for _ in range(3):
        expected = torch.log2(expected * 0.01 + 2)
    torch.testing.assert_close(kernel()[0], expected.sum(1), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("threads", [32, 64, 256, 512])
def test_gpu_consumer_probe_thread_domains(threads):
    import torch
    import tilelang
    from tilelang.tiletune.profiling.device_probes import reduction_primitive

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    kernel = tilelang.compile(reduction_primitive(0, 3, 1, threads), out_idx=[0], execution_backend="tvm_ffi")
    initial = 0.01 * (torch.arange(threads, device="cuda")[:, None] + torch.arange(8, device="cuda")[None, :] + 1)
    torch.testing.assert_close(kernel()[0], (initial + 0.0003).sum(1), rtol=1e-5, atol=1e-6)
