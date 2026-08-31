from __future__ import annotations

import tilelang.language as T
from tilelang import tvm
from tilelang.autotuner.filters import (
    AutotuneFilterConfig,
    AutotuneRuleContext,
    AutotuneVerifyRule,
    classify_kernel_filter_info,
    evaluate_post_compile_filter,
    evaluate_pre_compile_filter,
    extract_cuda_function_source,
    extract_cuda_kernel_filter_info,
    extract_launch_resource_info,
    extract_pre_compile_filter_info,
    iter_filter_rules,
    register_filter_rule,
    unregister_filter_rule,
)
from tilelang.autotuner.filters import LaunchResourceInfo
from tilelang.contrib.cuda_resource_info import KernelResourceUsage
from tilelang.engine.lower import device_codegen_without_compile, lower_to_host_device_ir
from tvm import tirx
from tvm.target import Target


SLOW_KERNEL_SOURCE = r"""
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
"""

ATTENTION_KERNEL_SOURCE = r"""
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
"""

QUANTIZED_GEMM_KERNEL_SOURCE = r"""
extern "C" __global__ void quant_kernel(const void* A, const void* B) {
  float Ct_local[128];
  half_t B_dequantize_local[192];
  for (int k = 0; k < 8; ++k) {
  }
}
"""

SPARSE_GEMM_KERNEL_SOURCE = r"""
extern "C" __global__ void sparse_kernel(const void* A, const void* BlockMask) {
  float C_local[128];
  if (((bool*)BlockMask)[0]) {
  }
}
"""


def test_extract_cuda_function_source_selects_named_kernel_from_grouped_source():
    source = extract_cuda_function_source(SLOW_KERNEL_SOURCE, "main_kernel")

    assert "float C_local[512]" in source
    assert "float C_local[32]" not in source


def test_cuda_filter_info_extracts_exact_features():
    info = extract_cuda_kernel_filter_info(
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
    assert info.detected_kernel_type == "dense_gemm"
    assert "uses_wgmma" in info.detected_kernel_traits
    assert "uses_tma" in info.detected_kernel_traits


def test_filter_rules_are_classified_by_layer():
    common_names = {rule.name for rule in iter_filter_rules(layer="common")}
    primitive_names = {rule.name for rule in iter_filter_rules(layer="primitive")}
    kernel_names = {rule.name for rule in iter_filter_rules(layer="kernel")}

    assert "common.spills" in common_names
    assert "common.local_memory" in common_names
    assert "primitive.wgmma_n" in primitive_names
    assert "primitive.tma_tiny_tile" in primitive_names
    assert "gemm.c_local" in kernel_names
    assert "attention.state_elements" in kernel_names


def test_filter_rule_registry_accepts_plugin_rule():
    class _TestPluginRule(AutotuneVerifyRule):
        name = "test.plugin_rule"
        layer = "kernel"
        finding_kind = "violation"

        def check(self, context: AutotuneRuleContext):
            return [{"reason": "test_plugin_rule", "function": context.info.function_name}]

    register_filter_rule(_TestPluginRule())
    try:
        decision = evaluate_post_compile_filter(
            launch_infos=[LaunchResourceInfo("plugin_kernel", block_dims=(128, 1, 1))],
            resource_usage={"plugin_kernel": KernelResourceUsage(n_spills=0, local_size_bytes=0)},
            kernel_source='extern "C" __global__ void plugin_kernel() {}',
            config={},
            filter_config=AutotuneFilterConfig(enabled=True),
        )
    finally:
        unregister_filter_rule("test.plugin_rule")

    assert decision.verdict == "reject"
    assert decision.details["violations"] == [
        {
            "reason": "test_plugin_rule",
            "function": "plugin_kernel",
            "kernel_type": "generic",
        }
    ]


def test_filter_rule_registry_accepts_trait_plugin_rule():
    class _TestTraitRule(AutotuneVerifyRule):
        name = "test.trait_rule"
        layer = "primitive"
        finding_kind = "violation"
        required_kernel_traits = frozenset({"has_custom_trait"})

        def check(self, context: AutotuneRuleContext):
            return [{"reason": "test_trait_rule", "function": context.info.function_name}]

    register_filter_rule(_TestTraitRule())
    try:
        decision = evaluate_post_compile_filter(
            launch_infos=[LaunchResourceInfo("trait_kernel", block_dims=(128, 1, 1))],
            resource_usage={"trait_kernel": KernelResourceUsage(n_spills=0, local_size_bytes=0)},
            kernel_source='extern "C" __global__ void trait_kernel() {}',
            config={},
            filter_config=AutotuneFilterConfig(
                enabled=True,
                check_spills=False,
                check_local_memory=False,
                check_c_local=False,
                check_output_elements_per_thread=False,
                check_tma_tiny_tile=False,
                check_wgmma_n=False,
                check_k_loop=False,
                kernel_traits=["has_custom_trait"],
            ),
        )
    finally:
        unregister_filter_rule("test.trait_rule")

    assert decision.verdict == "reject"
    assert decision.details["violations"] == [
        {
            "reason": "test_trait_rule",
            "function": "trait_kernel",
            "kernel_type": "generic",
        }
    ]
    assert decision.details["classifications"][0]["traits"] == ["has_custom_trait"]


def _make_wgmma_pre_compile_module(function_name="pre_compile_kernel", wgmma_prefix="m64n256k16"):
    wgmma_call = tirx.Call(
        "handle",
        tvm.ir.Op.get("tl.ptx_wgmma_ss"),
        [
            tirx.StringImm(wgmma_prefix),
            tirx.const(True, "bool"),
            tirx.const(True, "bool"),
            tirx.StringImm("float16"),
            tirx.StringImm("float16"),
            tirx.StringImm("float32"),
            tirx.Var("a_desc", "handle"),
            tirx.IntImm("int32", 0),
            tirx.Var("b_desc", "handle"),
            tirx.IntImm("int32", 0),
            tirx.Var("c_data", "handle"),
            tirx.IntImm("int32", 0),
            tirx.IntImm("int32", 1),
            tirx.const(True, "bool"),
            tirx.const(True, "bool"),
        ],
    )
    func = tirx.PrimFunc([], tirx.Evaluate(wgmma_call)).with_attr("global_symbol", function_name)
    return tvm.IRModule({function_name: func})


def test_pre_compile_info_extracts_resolved_wgmma_shape_before_codegen():
    info = extract_pre_compile_filter_info(
        function_name="pre_compile_kernel",
        device_mod=_make_wgmma_pre_compile_module(),
        launch_info=LaunchResourceInfo("pre_compile_kernel", block_dims=(128, 1, 1)),
        config={"block_M": 256, "block_N": 256, "thread_num": 128, "K": 4096, "block_K": 64},
    )

    assert info.source_available is False
    assert info.wgmma_shapes == [(64, 256, 16)]
    assert info.max_wgmma_n == 256
    assert info.output_elements_per_thread == 512
    assert info.max_k_loop_iterations == 64
    assert info.detected_kernel_type == "generic"
    assert "uses_wgmma" in info.detected_kernel_traits
    assert "uses_gemm" in info.detected_kernel_traits


def test_pre_compile_filter_rejects_large_wgmma_before_codegen():
    decision = evaluate_pre_compile_filter(
        launch_infos=[LaunchResourceInfo("pre_compile_kernel", block_dims=(128, 1, 1))],
        device_mod=_make_wgmma_pre_compile_module(),
        config={"block_M": 256, "block_N": 256, "thread_num": 128},
        filter_config=AutotuneFilterConfig(
            enabled=True,
            check_spills=False,
            check_local_memory=False,
            check_c_local=False,
            check_output_elements_per_thread=False,
            check_tma_tiny_tile=False,
            max_wgmma_n=128,
        ),
    )

    assert decision.stage == "pre_compile"
    assert decision.verdict == "reject"
    reasons = {advisory["reason"] for advisory in decision.details["advisories"]}
    assert "wgmma_n_over_limit" in reasons


def _make_dense_gemm_kernel(M=512, N=512, K=512, dtype=T.float16, accum_dtype=T.float32):
    def kernel(block_M=None, block_N=None, block_K=None, num_stages=1, thread_num=None):
        @T.prim_func
        def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                T.copy(C_local, C[by * block_M, bx * block_N])

        return main

    return kernel


def _filter_shapes(infos):
    return sorted(shape for info in infos for shape in info.wgmma_shapes)


def test_pre_compile_wgmma_matches_cuda_source_filter_before_nvcc():
    target = Target({"kind": "cuda", "arch": "sm_90"})
    kernel = _make_dense_gemm_kernel()
    configs = [
        {"block_M": 64, "block_N": 64, "block_K": 64, "num_stages": 1, "thread_num": 128},
        {"block_M": 64, "block_N": 128, "block_K": 64, "num_stages": 1, "thread_num": 128},
        {"block_M": 128, "block_N": 256, "block_K": 64, "num_stages": 2, "thread_num": 256},
    ]
    filter_config = AutotuneFilterConfig(
        enabled=True,
        kernel_type="dense_gemm",
        check_spills=False,
        check_local_memory=False,
        check_registers=False,
        check_c_local=False,
        check_output_elements_per_thread=False,
        check_tma_tiny_tile=False,
        check_k_loop=False,
        max_wgmma_n=64,
    )

    for config in configs:
        program = kernel(**config)
        with tvm.transform.PassContext(opt_level=3), target:
            _, device_mod, _, normalized_target, _ = lower_to_host_device_ir(program, target=target)
        launch_infos = extract_launch_resource_info(device_mod)

        pre_compile_infos = [
            extract_pre_compile_filter_info(
                function_name=launch.function_name,
                device_mod=device_mod,
                launch_info=launch,
                config=config,
            )
            for launch in launch_infos
        ]

        with tvm.transform.PassContext(opt_level=3), normalized_target:
            cuda_mod = device_codegen_without_compile(device_mod, normalized_target)
        cuda_source = cuda_mod.inspect_source()
        combined_pre_compile_infos = [
            extract_pre_compile_filter_info(
                function_name=launch.function_name,
                device_mod=device_mod,
                launch_info=launch,
                config=config,
                kernel_source=cuda_source,
            )
            for launch in launch_infos
        ]
        cuda_infos = [
            extract_cuda_kernel_filter_info(
                function_name=launch.function_name,
                kernel_source=cuda_source,
                launch_info=launch,
                raw_usage=KernelResourceUsage(),
                config=config,
            )
            for launch in launch_infos
        ]

        assert _filter_shapes(pre_compile_infos) == _filter_shapes(cuda_infos)
        assert _filter_shapes(combined_pre_compile_infos) == _filter_shapes(cuda_infos)

        pre_compile_decision = evaluate_pre_compile_filter(
            launch_infos=launch_infos,
            device_mod=device_mod,
            config=config,
            filter_config=filter_config,
            kernel_source=cuda_source,
        )

        assert pre_compile_decision.stage == "pre_compile"
        if any(shape[1] > 64 for shape in _filter_shapes(combined_pre_compile_infos)):
            assert pre_compile_decision.verdict == "reject"
            assert pre_compile_decision.reason == "filter_advisory_applied"
            reasons = {advisory["reason"] for advisory in pre_compile_decision.details["advisories"]}
            assert "wgmma_n_over_limit" in reasons
        else:
            assert pre_compile_decision.verdict == "keep"


def test_attention_filter_info_extracts_exact_fragment_state():
    info = extract_cuda_kernel_filter_info(
        function_name="attention_kernel",
        kernel_source=ATTENTION_KERNEL_SOURCE,
        launch_info=LaunchResourceInfo("attention_kernel", block_dims=(256, 1, 1)),
        raw_usage=KernelResourceUsage(n_regs=168, n_spills=12, local_size_bytes=48),
        config={"block_M": 128, "block_N": 256, "threads": 256, "num_stages": 1},
    )

    assert info.detected_kernel_type == "attention"
    assert "has_attention_state" in info.detected_kernel_traits
    assert "has_softmax" in info.detected_kernel_traits
    assert info.attention_score_elements_per_thread == 128
    assert info.attention_output_elements_per_thread == 64
    assert info.attention_softmax_elements_per_thread == 24
    assert info.attention_state_elements_per_thread == 216
    assert info.attention_cast_elements_per_thread == 128


def test_quantized_gemm_filter_info_extracts_dequant_state():
    info = extract_cuda_kernel_filter_info(
        function_name="quant_kernel",
        kernel_source=QUANTIZED_GEMM_KERNEL_SOURCE,
        launch_info=LaunchResourceInfo("quant_kernel", block_dims=(128, 1, 1)),
        raw_usage=KernelResourceUsage(n_regs=120),
        config={"block_M": 64, "block_N": 128, "threads": 128},
    )

    assert info.detected_kernel_type == "quantized_gemm"
    assert "has_dequant" in info.detected_kernel_traits
    assert info.c_local_floats == 128
    assert info.quant_dequant_elements_per_thread == 192


def test_sparse_gemm_filter_info_extracts_mask_signal():
    info = extract_cuda_kernel_filter_info(
        function_name="sparse_kernel",
        kernel_source=SPARSE_GEMM_KERNEL_SOURCE,
        launch_info=LaunchResourceInfo("sparse_kernel", block_dims=(128, 1, 1)),
        raw_usage=KernelResourceUsage(n_regs=120),
        config={"block_M": 64, "block_N": 128, "threads": 128},
    )

    assert info.detected_kernel_type == "sparse_gemm"
    assert "has_sparse_mask" in info.detected_kernel_traits
    assert info.sparse_mask_access_count > 0


def test_user_kernel_type_override_wins_over_detected_type():
    info = extract_cuda_kernel_filter_info(
        function_name="main_kernel",
        kernel_source=SLOW_KERNEL_SOURCE,
        launch_info=LaunchResourceInfo("main_kernel", block_dims=(128, 1, 1)),
        raw_usage=KernelResourceUsage(n_spills=224, local_size_bytes=0),
        config={"block_M": 64, "block_N": 64, "block_K": 64, "thread_num": 128},
    )
    classification = classify_kernel_filter_info(
        info,
        AutotuneFilterConfig(
            enabled=True,
            kernel_type="attention",
            kernel_traits=["user_hint"],
        ),
    )

    assert info.detected_kernel_type == "dense_gemm"
    assert classification.primary_kernel_type == "attention"
    assert classification.kernel_type_tags == ("attention",)
    assert "user_hint" in classification.traits


def test_user_kernel_type_override_dispatches_attention_rules():
    decision = evaluate_post_compile_filter(
        launch_infos=[LaunchResourceInfo("main_kernel", block_dims=(128, 1, 1))],
        resource_usage={"main_kernel": KernelResourceUsage(n_spills=224, local_size_bytes=0)},
        kernel_source=SLOW_KERNEL_SOURCE,
        config={"block_M": 64, "block_N": 64, "block_K": 64, "thread_num": 128},
        filter_config=AutotuneFilterConfig(
            enabled=True,
            kernel_type="attention",
            check_attention_local_memory=False,
        ),
    )

    reasons = {violation["reason"] for violation in decision.details["violations"]}
    assert decision.verdict == "reject"
    assert "attention_spills_over_limit" in reasons
    assert "spills_over_limit" not in reasons
    assert decision.details["classifications"][0]["primary_kernel_type"] == "attention"


def test_filter_rejects_enabled_targets():
    config = {"block_M": 256, "block_N": 256, "thread_num": 128, "num_stages": 0}
    filter_config = AutotuneFilterConfig(enabled=True)

    pre_compile_decision = evaluate_pre_compile_filter(
        launch_infos=[LaunchResourceInfo("main_kernel", block_dims=(128, 1, 1))],
        device_mod=tvm.IRModule({}),
        config=config,
        filter_config=filter_config,
        kernel_source=SLOW_KERNEL_SOURCE,
    )
    post_compile_decision = evaluate_post_compile_filter(
        launch_infos=[LaunchResourceInfo("main_kernel", block_dims=(128, 1, 1))],
        resource_usage={
            "main_kernel": KernelResourceUsage(
                n_regs=255,
                n_spills=1094,
                local_size_bytes=2600,
            )
        },
        kernel_source=SLOW_KERNEL_SOURCE,
        config=config,
        filter_config=filter_config,
    )

    assert pre_compile_decision.stage == "pre_compile"
    assert pre_compile_decision.verdict == "reject"
    pre_compile_reasons = {advisory["reason"] for advisory in pre_compile_decision.details["advisories"]}
    assert "c_local_floats_over_limit" in pre_compile_reasons
    assert "output_elements_per_thread_over_limit" in pre_compile_reasons

    assert post_compile_decision.stage == "post_compile"
    assert post_compile_decision.verdict == "reject"
    post_compile_reasons = {violation["reason"] for violation in post_compile_decision.details["violations"]}
    assert "spills_over_limit" in post_compile_reasons
    assert "local_memory_over_limit" in post_compile_reasons
    assert "c_local_floats_over_limit" not in post_compile_reasons
    assert "output_elements_per_thread_over_limit" not in post_compile_reasons


def test_attention_filter_profile_uses_thresholded_spill_and_state_targets():
    config = {"block_M": 128, "block_N": 256, "threads": 256, "num_stages": 1}
    filter_config = AutotuneFilterConfig(enabled=True, action="report")

    pre_compile_decision = evaluate_pre_compile_filter(
        launch_infos=[LaunchResourceInfo("attention_kernel", block_dims=(256, 1, 1))],
        device_mod=tvm.IRModule({}),
        config=config,
        filter_config=filter_config,
        kernel_source=ATTENTION_KERNEL_SOURCE,
    )
    post_compile_decision = evaluate_post_compile_filter(
        launch_infos=[LaunchResourceInfo("attention_kernel", block_dims=(256, 1, 1))],
        resource_usage={
            "attention_kernel": KernelResourceUsage(
                n_regs=168,
                n_spills=12,
                local_size_bytes=48,
            )
        },
        kernel_source=ATTENTION_KERNEL_SOURCE,
        config=config,
        filter_config=filter_config,
    )

    assert pre_compile_decision.verdict == "keep"
    assert pre_compile_decision.reason == "filter_advisory_report_only"
    pre_compile_advisories = {advisory["reason"] for advisory in pre_compile_decision.details["advisories"]}
    assert "wgmma_n_over_limit" in pre_compile_advisories

    assert post_compile_decision.verdict == "keep"
    assert post_compile_decision.reason == "filter_targets_passed"
    assert not post_compile_decision.details["violations"]
    assert not post_compile_decision.details["advisories"]


def test_attention_filter_profile_rejects_large_state_and_large_spills():
    config = {"block_M": 256, "block_N": 256, "threads": 128, "num_stages": 1}
    kernel_source = ATTENTION_KERNEL_SOURCE.replace("float acc_s[128];", "float acc_s[512];")
    filter_config = AutotuneFilterConfig(enabled=True, kernel_type="attention")

    pre_compile_decision = evaluate_pre_compile_filter(
        launch_infos=[LaunchResourceInfo("attention_kernel", block_dims=(128, 1, 1))],
        device_mod=tvm.IRModule({}),
        config=config,
        filter_config=filter_config,
        kernel_source=kernel_source,
    )
    post_compile_decision = evaluate_post_compile_filter(
        launch_infos=[LaunchResourceInfo("attention_kernel", block_dims=(128, 1, 1))],
        resource_usage={
            "attention_kernel": KernelResourceUsage(
                n_regs=240,
                n_spills=224,
                local_size_bytes=384,
            )
        },
        kernel_source=kernel_source,
        config=config,
        filter_config=filter_config,
    )

    assert pre_compile_decision.verdict == "reject"
    assert pre_compile_decision.reason == "filter_advisory_applied"
    pre_compile_reasons = {advisory["reason"] for advisory in pre_compile_decision.details["advisories"]}
    assert "attention_state_elements_per_thread_over_limit" in pre_compile_reasons

    assert post_compile_decision.verdict == "reject"
    post_compile_reasons = {violation["reason"] for violation in post_compile_decision.details["violations"]}
    assert "attention_spills_over_limit" in post_compile_reasons
    assert "attention_local_memory_over_limit" in post_compile_reasons
    assert "attention_state_elements_per_thread_over_limit" not in post_compile_reasons


def test_quantized_gemm_profile_reports_dequant_advisory():
    decision = evaluate_pre_compile_filter(
        launch_infos=[LaunchResourceInfo("quant_kernel", block_dims=(128, 1, 1))],
        device_mod=tvm.IRModule({}),
        config={"block_M": 64, "block_N": 128, "threads": 128},
        filter_config=AutotuneFilterConfig(
            enabled=True,
            action="report",
            kernel_type="quantized_gemm",
            check_tma_tiny_tile=False,
        ),
        kernel_source=QUANTIZED_GEMM_KERNEL_SOURCE,
    )

    assert decision.verdict == "keep"
    assert decision.reason == "filter_advisory_report_only"
    advisory_reasons = {advisory["reason"] for advisory in decision.details["advisories"]}
    assert "quant_dequant_elements_per_thread_over_limit" in advisory_reasons


def test_sparse_gemm_profile_can_reject_missing_mask_when_requested():
    decision = evaluate_pre_compile_filter(
        launch_infos=[LaunchResourceInfo("sparse_kernel", block_dims=(128, 1, 1))],
        device_mod=tvm.IRModule({}),
        config={"block_M": 64, "block_N": 128, "threads": 128},
        filter_config=AutotuneFilterConfig(
            enabled=True,
            kernel_type="sparse_gemm",
            check_sparse_mask=True,
            check_tma_tiny_tile=False,
        ),
        kernel_source='extern "C" __global__ void sparse_kernel() { float C_local[128]; }',
    )

    assert decision.verdict == "reject"
    reasons = {violation["reason"] for violation in decision.details["violations"]}
    assert "sparse_mask_not_detected" in reasons


def test_filter_targets_can_be_disabled_independently():
    decision = evaluate_post_compile_filter(
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
        filter_config=AutotuneFilterConfig(
            enabled=True,
            check_spills=False,
            check_local_memory=False,
            check_c_local=False,
            check_output_elements_per_thread=False,
            check_tma_tiny_tile=False,
            check_wgmma_n=False,
            check_k_loop=False,
        ),
    )

    assert decision.verdict == "keep"
    assert decision.reason == "filter_targets_passed"


def test_filter_report_action_keeps_advisory_findings():
    decision = evaluate_pre_compile_filter(
        launch_infos=[LaunchResourceInfo("main_kernel", block_dims=(128, 1, 1))],
        device_mod=tvm.IRModule({}),
        config={"block_M": 256, "block_N": 256, "thread_num": 128, "num_stages": 0},
        filter_config=AutotuneFilterConfig(
            enabled=True,
            action="report",
            check_spills=False,
            check_local_memory=False,
            check_c_local=False,
            check_output_elements_per_thread=False,
            check_tma_tiny_tile=False,
        ),
        kernel_source=SLOW_KERNEL_SOURCE,
    )

    assert decision.verdict == "keep"
    assert decision.reason == "filter_advisory_report_only"
    reasons = {advisory["reason"] for advisory in decision.details["advisories"]}
    assert "wgmma_n_over_limit" in reasons
    assert "k_loop_iterations_over_limit" in reasons


def test_filter_strict_wgmma_and_k_loop_can_reject():
    decision = evaluate_pre_compile_filter(
        launch_infos=[LaunchResourceInfo("main_kernel", block_dims=(128, 1, 1))],
        device_mod=tvm.IRModule({}),
        config={"block_M": 256, "block_N": 256, "thread_num": 128, "num_stages": 0},
        filter_config=AutotuneFilterConfig(
            enabled=True,
            check_spills=False,
            check_local_memory=False,
            check_c_local=False,
            check_output_elements_per_thread=False,
            check_tma_tiny_tile=False,
            max_wgmma_n=128,
            max_k_loop_iterations=64,
        ),
        kernel_source=SLOW_KERNEL_SOURCE,
    )

    assert decision.verdict == "reject"
    assert decision.reason == "filter_advisory_applied"
    reasons = {advisory["reason"] for advisory in decision.details["advisories"]}
    assert "wgmma_n_over_limit" in reasons
    assert "k_loop_iterations_over_limit" in reasons


def test_filter_report_action_still_rejects_hard_violations():
    decision = evaluate_post_compile_filter(
        launch_infos=[LaunchResourceInfo("main_kernel", block_dims=(128, 1, 1))],
        resource_usage={"main_kernel": KernelResourceUsage(n_spills=1)},
        kernel_source=SLOW_KERNEL_SOURCE,
        config={"block_M": 256, "block_N": 256, "thread_num": 128},
        filter_config=AutotuneFilterConfig(enabled=True, action="report"),
    )

    assert decision.verdict == "reject"
    assert decision.reason == "filter_hard_violation"
    assert decision.details["violations"]
