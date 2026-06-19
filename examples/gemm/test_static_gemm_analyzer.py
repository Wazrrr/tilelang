from __future__ import annotations

from types import SimpleNamespace

from static_gemm_analyzer import (
    GemmConfig,
    analyze_cuda_source,
    estimate_static_resources,
    parse_ptxas_output,
)


def _fake_h200_arch():
    return SimpleNamespace(
        name="NVIDIA H200",
        sm_version=90,
        compute_max_core=132,
        warp_size=32,
        smem_cap=228 * 1024,
        max_smem_usage=228 * 1024,
        reg_cap=65536,
        target=SimpleNamespace(attrs={"arch": "sm_90"}),
    )


def test_parse_ptxas_output():
    output = """
ptxas info    : Compiling entry function '_Z4mainv' for 'sm_90'
ptxas info    : Function properties for _Z4mainv
    0 bytes stack frame, 8 bytes spill stores, 16 bytes spill loads
ptxas info    : Used 96 registers, 2048 bytes smem, 392 bytes cmem[0]
"""
    info = parse_ptxas_output(output)

    assert info.registers_per_thread == 96
    assert info.smem_bytes == 2048
    assert info.cmem_bytes == 392
    assert info.spill_bytes == 24
    assert info.has_metadata


def test_static_resource_estimate_stage_scales_shared_memory():
    cfg_stage_1 = GemmConfig(block_M=128, block_N=128, block_K=64, num_stages=1, thread_num=256, enable_rasteration=True)
    cfg_stage_3 = GemmConfig(block_M=128, block_N=128, block_K=64, num_stages=3, thread_num=256, enable_rasteration=True)

    stage_1 = estimate_static_resources(cfg_stage_1, M=4096, N=4096, K=4096)
    stage_3 = estimate_static_resources(cfg_stage_3, M=4096, N=4096, K=4096)

    assert stage_3.pipeline_shared_bytes_estimate == stage_1.pipeline_shared_bytes_estimate * 3
    assert stage_3.shared_bytes_estimate > stage_1.shared_bytes_estimate
    assert stage_1.registers_per_thread_estimate > stage_1.accumulator_registers_per_thread_estimate


def test_analyze_cuda_source_rejects_spilling_config():
    cfg = GemmConfig(block_M=128, block_N=256, block_K=64, num_stages=3, thread_num=256, enable_rasteration=True)
    source = 'extern "C" __global__ void __launch_bounds__(256) main_kernel() { asm("wgmma.mma_async"); }'
    ptxas = "ptxas info    : Used 128 registers, 0 bytes smem\n0 bytes stack frame, 4 bytes spill stores, 0 bytes spill loads"

    report = analyze_cuda_source(source, cfg, _fake_h200_arch(), M=4096, N=4096, K=4096, ptxas_output=ptxas)

    assert report.verdict == "reject"
    assert "register_spill_risk" in report.reasons
    assert report.source.launch_bounds_threads == 256
    assert report.source.wgmma_ops == 1


def test_analyze_cuda_source_keeps_resource_fit_config():
    cfg = GemmConfig(block_M=128, block_N=128, block_K=64, num_stages=2, thread_num=256, enable_rasteration=True)
    source = 'extern "C" __global__ void __launch_bounds__(256) main_kernel() { asm("wgmma.mma_async"); }'
    ptxas = "ptxas info    : Used 80 registers, 0 bytes smem\n0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads"

    report = analyze_cuda_source(source, cfg, _fake_h200_arch(), M=4096, N=4096, K=4096, ptxas_output=ptxas)

    assert report.verdict == "keep"
    assert report.active_blocks_per_sm_estimate > 0
    assert report.score > 0
