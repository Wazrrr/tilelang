"""Hopper instruction and scheduling contracts for the experiment kernels."""

import pytest

from experiments.common.kernels import make_case
from experiments.kda.cases import cases
from experiments.common.spec import Workload
from tilelang.tiletune import analyze_prim_func
from test_device_profile import profile
from test_portability import LIMITS

WORKLOADS = cases(True) + [
    Workload("kda_tail_regression", "kda_chunk_o", dict(batch=1, heads=4, sequence=768, dim=96, value_dim=80, chunk_size=48))
]


@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda w: w.name)
@pytest.mark.parametrize("stages", [0, 2])
def test_kda_hopper_uses_the_compiler_instruction_and_gate_consumers(workload, stages):
    case = make_case(workload)
    func = case.build(block_DK=32, block_DV=64, num_stages=stages, threads=128)
    before = func.script()
    rates = dict(profile(), gemm_instruction_rates={"cuda.mma": dict(gemm_flops_per_cycle=512)})
    result = analyze_prim_func(
        func,
        dict(ranking_metric="pipeline_time", performance_model=rates),
        target=dict(kind="cuda", arch="sm_90a"),
        device_limits=LIMITS,
    )
    assert func.script() == before
    pipeline = result["modules"]["pipeline_overlap"]
    if stages and workload.parameters["value_dim"] % 64:
        # Region-varying TMA traffic needs a verified per-region schedule.
        # Keep this explicit rather than scoring a uniform padded transfer.
        assert result["tile_cost"]["score"] is None
        assert any(d["reason"] == "independent pipeline has no verified compiler plan" for d in pipeline["diagnostics"])
    else:
        assert not pipeline["unknown"], pipeline["unknown"]
        assert result["tile_cost"]["score"] is not None
    assert result["specialization"]["name"] == "kda_chunk_o"
    matrices = [p for p in pipeline["phases"] if p["work"]["gemm_flops"]]
    instruction = "cuda.wgmma" if workload.parameters["chunk_size"] == 64 else "cuda.mma"
    assert [p["compute_participants"]["instruction"] for p in matrices] == [instruction] * 2
    assert [p["inside_loop"] for p in matrices] == [True, False]
    assert sum(p["work"]["exp_ops"] for p in pipeline["phases"]) == workload.parameters["chunk_size"] * 32
    if stages:
        ws = result["pressure"]["warp_specialization"]
        assert ws["status"] == "predicted" and ws["launch_threads"] == 256
        # Q and gate tiles can be reused after gating; the state tile stays
        # live through the first GEMM. This is not a single GEMM barrier group.
        copies = pipeline["producer_buffers"]
        assert copies[0]["last_consumer"] == copies[1]["last_consumer"] < copies[2]["last_consumer"]


def test_phase_service_selects_instruction_rates_without_dtype_fallback():
    from tiletune_core.compute import estimate_phase_cycles

    rates = dict(profile(), gemm_instruction_rates={"cuda.mma": dict(gemm_flops_per_cycle=512)})
    phase = dict(
        work=dict(gemm_flops=8192, shared_bytes=0, elementwise_ops=0, exp_ops=0, reduction_ops=0),
        compute_participants=dict(rates["gemm_signature"], instruction="cuda.mma", precision="predicted"),
    )
    assert estimate_phase_cycles(phase, rates, 2) == 32
    assert estimate_phase_cycles(phase, profile(), 2) is None
    phase["compute_participants"]["a_dtype"] = "float8_e4m3fn"
    assert estimate_phase_cycles(phase, rates, 2) is None


@pytest.mark.parametrize("row", [{}, {"gemm_flops_per_cycle": 0}, {"gemm_flops_per_cycle": 512, "exp_ops_per_cycle": 64}])
def test_instruction_rates_reject_missing_or_invalid_matrix_measurements(row):
    from tilelang.tiletune import TileTuneConfig

    with pytest.raises(ValueError):
        TileTuneConfig(performance_model=dict(profile(), gemm_instruction_rates={"cuda.mma": row}))


@pytest.mark.parametrize("dtype", ["float16", "float8_e4m3fn", "float8_e5m2"])
def test_hopper_mma_profile_probe_cross_compiles(dtype):
    import tilelang
    from tilelang.tiletune.profiling.device_probes import tensor_core
    from tilelang.transform import PassContext
    from tvm.target import Target

    with Target(dict(kind="cuda", arch="sm_90a")) as target, PassContext(config={"tl.disable_wgmma": True}):
        artifact = tilelang.lower(tensor_core(dtype, "float32", 3, 1, 128), target=target, enable_device_compile=True)
    assert "tl::mma_sync" in artifact.kernel_source
    assert "tl::wgmma" not in artifact.kernel_source


@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda w: w.name)
def test_kda_hopper_cross_compile_matches_instruction_and_partition(workload):
    import tilelang
    from tvm.target import Target

    func = make_case(workload).build(block_DK=32, block_DV=64, num_stages=2, threads=128)
    with Target(dict(kind="cuda", arch="sm_90a")) as target:
        artifact = tilelang.lower(func, target=target, enable_device_compile=True)
    source = artifact.kernel_source
    assert "__launch_bounds__(256" in source
    instruction = "tl::wgmma_ss" if workload.parameters["chunk_size"] == 64 else "tl::mma_sync"
    assert instruction in source
