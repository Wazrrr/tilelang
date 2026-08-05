from __future__ import annotations

from tilelang.autotuner.quality_filter import (
    AutotuneQualityFilterConfig,
    evaluate_post_compile_quality_filter,
    extract_cuda_function_source,
    extract_cuda_kernel_quality_info,
)
from tilelang.autotuner.resource_filter import LaunchResourceInfo
from tilelang.contrib.cuda_resource_info import KernelResourceUsage


SLOW_KERNEL_SOURCE = r'''
extern "C" __global__ void other_kernel() {
  float C_local[32];
}

extern "C" __global__ void main_kernel(const void* A, const void* B) {
  extern __shared__ __align__(1024) unsigned char buf_dyn_shmem[];
  float C_local[512];
  for (int k = 0; k < 128; ++k) {
    __syncthreads();
    tl::tma_load(A_desc, mbarrier[(k % 3)], buf_dyn_shmem, k * 32, 0);
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      for (int ki = 0; ki < 2; ++ki) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                     tl::DataType::kFloat32, 64, 256, 16, false, false, 1, 1>(
            0, 0, ((uint32_t*)(C_local + (i_3 * 128))), 1);
      }
    }
  }
  for (int i_4 = 0; i_4 < 64; ++i_4) {
    tl::ptx_stmatrix_m8n8_x4(buf_dyn_shmem, 0, 0, 0, 0);
  }
  tl::tma_store(C_desc, buf_dyn_shmem, 0, 0);
  tl::tma_store(C_desc, buf_dyn_shmem, 64, 0);
  tl::tma_store(C_desc, buf_dyn_shmem, 128, 0);
  tl::tma_store(C_desc, buf_dyn_shmem, 192, 0);
}
'''


def test_extract_cuda_function_source_selects_named_kernel_from_grouped_source():
    source = extract_cuda_function_source(SLOW_KERNEL_SOURCE, "main_kernel")

    assert "float C_local[512]" in source
    assert "float C_local[32]" not in source


def test_cuda_quality_info_extracts_exact_features():
    info = extract_cuda_kernel_quality_info(
        function_name="main_kernel",
        kernel_source=SLOW_KERNEL_SOURCE,
        launch_info=LaunchResourceInfo("main_kernel", block_dims=(128, 1, 1)),
        raw_usage=KernelResourceUsage(
            n_regs=255,
            n_spills=1094,
            local_size_bytes=2600,
            extra={"spill_stores_bytes": 4376, "spill_loads_bytes": 3964},
        ),
        config={"block_M": 256, "block_N": 256, "thread_num": 128, "num_stages": 0},
    )

    assert info.c_local_floats == 512
    assert info.max_wgmma_n == 256
    assert info.max_k_loop_iterations == 128
    assert info.output_elements_per_thread == 512
    assert info.tma_store_count == 4
    assert info.n_spills == 1094
    assert info.local_size_bytes == 2600


def test_quality_filter_rejects_enabled_targets():
    decision = evaluate_post_compile_quality_filter(
        launch_infos=[LaunchResourceInfo("main_kernel", block_dims=(128, 1, 1))],
        resource_usage={
            "main_kernel": KernelResourceUsage(
                n_regs=255,
                n_spills=1094,
                local_size_bytes=2600,
            )
        },
        kernel_source=SLOW_KERNEL_SOURCE,
        config={"block_M": 256, "block_N": 256, "thread_num": 128, "num_stages": 0},
        quality_config=AutotuneQualityFilterConfig(enabled=True),
    )

    assert decision.verdict == "reject"
    reasons = {violation["reason"] for violation in decision.details["violations"]}
    assert "spills_over_quality_limit" in reasons
    assert "local_memory_over_quality_limit" in reasons
    assert "c_local_floats_over_quality_limit" in reasons
    assert "output_elements_per_thread_over_quality_limit" in reasons


def test_quality_filter_targets_can_be_disabled_independently():
    decision = evaluate_post_compile_quality_filter(
        launch_infos=[LaunchResourceInfo("main_kernel", block_dims=(128, 1, 1))],
        resource_usage={
            "main_kernel": KernelResourceUsage(
                n_regs=255,
                n_spills=1094,
                local_size_bytes=2600,
            )
        },
        kernel_source=SLOW_KERNEL_SOURCE,
        config={"block_M": 256, "block_N": 256, "thread_num": 128, "num_stages": 0},
        quality_config=AutotuneQualityFilterConfig(
            enabled=True,
            check_spills=False,
            check_local_memory=False,
            check_c_local=False,
            check_output_elements_per_thread=False,
            check_tma_tiny_tile=False,
            check_wgmma_n_advisory=False,
            check_k_loop_advisory=False,
        ),
    )

    assert decision.verdict == "keep"
    assert decision.reason == "quality_targets_passed"


def test_quality_filter_advisories_do_not_reject():
    decision = evaluate_post_compile_quality_filter(
        launch_infos=[LaunchResourceInfo("main_kernel", block_dims=(128, 1, 1))],
        resource_usage={"main_kernel": KernelResourceUsage(n_spills=0, local_size_bytes=0)},
        kernel_source=SLOW_KERNEL_SOURCE,
        config={"block_M": 256, "block_N": 256, "thread_num": 128, "num_stages": 0},
        quality_config=AutotuneQualityFilterConfig(
            enabled=True,
            check_spills=False,
            check_local_memory=False,
            check_c_local=False,
            check_output_elements_per_thread=False,
            check_tma_tiny_tile=False,
        ),
    )

    assert decision.verdict == "keep"
    assert decision.reason == "quality_advisory_only"
    reasons = {advisory["reason"] for advisory in decision.details["advisories"]}
    assert "wgmma_n_over_advisory_limit" in reasons
    assert "k_loop_iterations_over_advisory_limit" in reasons


def test_quality_filter_strict_wgmma_and_k_loop_can_reject():
    decision = evaluate_post_compile_quality_filter(
        launch_infos=[LaunchResourceInfo("main_kernel", block_dims=(128, 1, 1))],
        resource_usage={"main_kernel": KernelResourceUsage(n_spills=0, local_size_bytes=0)},
        kernel_source=SLOW_KERNEL_SOURCE,
        config={"block_M": 256, "block_N": 256, "thread_num": 128, "num_stages": 0},
        quality_config=AutotuneQualityFilterConfig(
            enabled=True,
            check_spills=False,
            check_local_memory=False,
            check_c_local=False,
            check_output_elements_per_thread=False,
            check_tma_tiny_tile=False,
            max_wgmma_n=128,
            max_k_loop_iterations=64,
        ),
    )

    assert decision.verdict == "reject"
    reasons = {violation["reason"] for violation in decision.details["violations"]}
    assert "wgmma_n_over_quality_limit" in reasons
    assert "k_loop_iterations_over_quality_limit" in reasons


def test_quality_filter_report_action_keeps_with_violations():
    decision = evaluate_post_compile_quality_filter(
        launch_infos=[LaunchResourceInfo("main_kernel", block_dims=(128, 1, 1))],
        resource_usage={"main_kernel": KernelResourceUsage(n_spills=1)},
        kernel_source=SLOW_KERNEL_SOURCE,
        config={"block_M": 256, "block_N": 256, "thread_num": 128},
        quality_config=AutotuneQualityFilterConfig(enabled=True, action="report"),
    )

    assert decision.verdict == "keep"
    assert decision.reason == "quality_report_only"
    assert decision.details["violations"]
