from __future__ import annotations

import pytest

from static_gemm_analyzer import (
    CudaDeviceLimits,
    KernelResourceUsage,
    LaunchResourceInfo,
    analyze_compiled_resources,
    analyze_config_space,
    analyze_launch_resources,
    estimate_registers_from_device_code,
    parse_ptxas_output,
)


BASE_CONFIG = {
    "block_M": 64,
    "block_N": 64,
    "block_K": 64,
    "num_stages": 2,
    "thread_num": 256,
}


def _limits() -> CudaDeviceLimits:
    return CudaDeviceLimits(
        max_threads_per_block=1024,
        max_block_dims=(1024, 1024, 64),
        max_grid_dims=(2**31 - 1, 65535, 65535),
        max_shared_memory_per_block=49152,
        max_shared_memory_per_block_optin=228 * 1024,
        max_registers_per_block=65536,
        cooperative_launch=True,
    )


def test_source_register_estimation_is_rejected():
    with pytest.raises(RuntimeError, match="after NVCC/PTXAS compilation"):
        estimate_registers_from_device_code("float local[32];")


def test_parse_ptxas_output_reports_exact_registers_and_smem():
    output = """
ptxas info    : Compiling entry function 'main_kernel' for 'sm_90'
ptxas info    : Function properties for main_kernel
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 80 registers, 49152 bytes smem, 392 bytes cmem[0]
"""
    usage = parse_ptxas_output(output)

    assert usage["main_kernel"] == KernelResourceUsage(
        n_regs=80,
        static_smem_bytes=49152,
        const_size_bytes=392,
        extra={"cmem[0]": 392, "spill_stores_bytes": 0, "spill_loads_bytes": 0},
    )


def test_pre_compile_rejects_only_exact_launch_overbooking():
    report = analyze_launch_resources(
        BASE_CONFIG,
        [LaunchResourceInfo("main_kernel", block_dims=(2048, 1, 1), dynamic_smem_bytes=0)],
        _limits(),
    )

    assert report.verdict == "reject"
    assert report.reason == "threads_per_block_over_limit"


def test_pre_compile_keeps_unknown_symbolic_values():
    report = analyze_launch_resources(
        BASE_CONFIG,
        [LaunchResourceInfo("main_kernel", block_dims=(None, 1, 1), dynamic_smem_bytes=None)],
        _limits(),
    )

    assert report.verdict == "keep"


def test_post_compile_rejects_registers_per_block_overbooking():
    output = """
ptxas info    : Compiling entry function 'main_kernel' for 'sm_90'
ptxas info    : Used 256 registers
"""
    report = analyze_compiled_resources(
        BASE_CONFIG,
        [LaunchResourceInfo("main_kernel", block_dims=(512, 1, 1), dynamic_smem_bytes=0)],
        output,
        _limits(),
    )

    assert report.verdict == "reject"
    assert report.reason == "registers_per_block_over_limit"


def test_config_space_returns_only_exactly_kept_configs():
    good = {**BASE_CONFIG, "thread_num": 256}
    bad = {**BASE_CONFIG, "thread_num": 2048}
    summary = analyze_config_space(
        [good, bad],
        [
            [LaunchResourceInfo("main_kernel", block_dims=(256, 1, 1), dynamic_smem_bytes=0)],
            [LaunchResourceInfo("main_kernel", block_dims=(2048, 1, 1), dynamic_smem_bytes=0)],
        ],
        _limits(),
    )

    assert summary.selected_configs() == [good]
    assert len(summary.rejected_reports) == 1
