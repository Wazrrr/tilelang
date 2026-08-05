from __future__ import annotations

from tilelang.autotuner.filters import (
    AutotuneQualityFilterConfig,
    evaluate_post_compile_quality_filter,
    extract_cuda_function_source,
    extract_cuda_kernel_quality_info,
)
from tilelang.autotuner.filters import LaunchResourceInfo
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

ATTENTION_KERNEL_SOURCE = r'''
extern "C" __global__ void attention_kernel(const void* Q, const void* K, const void* V) {
  float acc_o[64];
  float logsum[4];
  float scores_max[4];
  float acc_s[128];
  float scores_max_prev[4];
  float scores_max_clear[4];
  float scores_scale[4];
  float scores_sum[4];
  half_t acc_s_cast[128];
  for (int k = 0; k < 4; ++k) {
    tl::tma_load(Q_desc, mbarrier[(k % 2)], smem, k * 64, 0);
    tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16,
                 tl::DataType::kFloat32, 64, 256, 16, false, false, 1, 1>(
        0, 0, ((uint32_t*)(acc_s + 0)), 1);
    tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16,
                 tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
        reinterpret_cast<const uint32_t*>(acc_s_cast + 0), 0, reinterpret_cast<uint32_t*>(acc_o + 0), 1);
  }
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


def test_attention_quality_info_extracts_exact_fragment_state():
    info = extract_cuda_kernel_quality_info(
        function_name="attention_kernel",
        kernel_source=ATTENTION_KERNEL_SOURCE,
        launch_info=LaunchResourceInfo("attention_kernel", block_dims=(256, 1, 1)),
        raw_usage=KernelResourceUsage(n_regs=168, n_spills=12, local_size_bytes=48),
        config={"block_M": 128, "block_N": 256, "threads": 256, "num_stages": 1},
    )

    assert info.detected_kernel_type == "attention"
    assert info.attention_score_elements_per_thread == 128
    assert info.attention_output_elements_per_thread == 64
    assert info.attention_softmax_elements_per_thread == 24
    assert info.attention_state_elements_per_thread == 216
    assert info.attention_cast_elements_per_thread == 128


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


def test_attention_quality_profile_uses_thresholded_spill_and_state_targets():
    decision = evaluate_post_compile_quality_filter(
        launch_infos=[LaunchResourceInfo("attention_kernel", block_dims=(256, 1, 1))],
        resource_usage={
            "attention_kernel": KernelResourceUsage(
                n_regs=168,
                n_spills=12,
                local_size_bytes=48,
            )
        },
        kernel_source=ATTENTION_KERNEL_SOURCE,
        config={"block_M": 128, "block_N": 256, "threads": 256, "num_stages": 1},
        quality_config=AutotuneQualityFilterConfig(enabled=True),
    )

    assert decision.verdict == "keep"
    assert decision.reason == "quality_advisory_only"
    assert not decision.details["violations"]
    advisory_reasons = {advisory["reason"] for advisory in decision.details["advisories"]}
    assert "attention_spills_over_advisory_limit" in advisory_reasons
    assert "attention_local_memory_over_advisory_limit" in advisory_reasons
    assert "wgmma_n_over_advisory_limit" in advisory_reasons


def test_attention_quality_profile_rejects_large_state_and_large_spills():
    decision = evaluate_post_compile_quality_filter(
        launch_infos=[LaunchResourceInfo("attention_kernel", block_dims=(128, 1, 1))],
        resource_usage={
            "attention_kernel": KernelResourceUsage(
                n_regs=240,
                n_spills=224,
                local_size_bytes=384,
            )
        },
        kernel_source=ATTENTION_KERNEL_SOURCE.replace("float acc_s[128];", "float acc_s[512];"),
        config={"block_M": 256, "block_N": 256, "threads": 128, "num_stages": 1},
        quality_config=AutotuneQualityFilterConfig(enabled=True, kernel_type="attention"),
    )

    assert decision.verdict == "reject"
    reasons = {violation["reason"] for violation in decision.details["violations"]}
    assert "attention_spills_over_quality_limit" in reasons
    assert "attention_local_memory_over_quality_limit" in reasons
    assert "attention_state_elements_per_thread_over_quality_limit" in reasons


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
