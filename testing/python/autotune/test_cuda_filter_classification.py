from __future__ import annotations

import re
from pathlib import Path

import pytest

from tilelang.autotuner.filters import (
    AutotuneFilterConfig,
    LaunchResourceInfo,
    classify_kernel_filter_info,
    extract_cuda_kernel_filter_info,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_AUTOTUNE_ENTRY_RE = re.compile(r"@(?:tilelang\.|tl\.)?autotune\b|AutoTuner\.from_kernel")

EXPECTED_AUTOTUNE_EXAMPLE_FAMILIES = {
    "examples/amd": "flash_attention",
    "examples/attention_sink": "attention_sink",
    "examples/blocksparse_gemm": "sparse_gemm",
    "examples/convolution": "conv",
    "examples/deepseek_mhc": "attention",
    "examples/deepseek_mla": "mla",
    "examples/deepseek_nsa": "block_sparse_attention",
    "examples/dequantize_gemm": "quantized_gemm",
    "examples/flash_attention": "flash_attention",
    "examples/flash_decoding": "flash_decoding",
    "examples/gdn": "linear_attention",
    "examples/gemm": "dense_gemm",
    "examples/gemm_fp8": "quantized_gemm",
    "examples/gemv": "gemv",
    "examples/kda": "linear_attention",
    "examples/linear_attention": "linear_attention",
    "examples/topk": "topk",
}
KNOWN_AUTOTUNE_FAMILIES = {
    "attention",
    "attention_sink",
    "block_sparse_attention",
    "conv",
    "dense_gemm",
    "flash_attention",
    "flash_decoding",
    "gemv",
    "linear_attention",
    "mla",
    "quantized_gemm",
    "sparse_gemm",
    "topk",
}

WGMMA_CALL = """
  tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16,
               tl::DataType::kFloat32, 64, 128, 16, false, false, 1, 1>(
      0, 0, reinterpret_cast<uint32_t*>(C_local + 0), 1);
"""


def _classify_source(
    function_name: str,
    source: str,
    *,
    config: dict[str, int] | None = None,
    filter_config: AutotuneFilterConfig | None = None,
):
    info = extract_cuda_kernel_filter_info(
        function_name=function_name,
        kernel_source=source,
        launch_info=LaunchResourceInfo(function_name, block_dims=(128, 1, 1)),
        raw_usage=None,
        config=config or {},
    )
    classification = classify_kernel_filter_info(info, filter_config or AutotuneFilterConfig(enabled=True))
    return info, classification


@pytest.mark.parametrize(
    ("function_name", "source", "config", "expected_type", "expected_tags", "expected_traits"),
    [
        (
            "dense_matmul_kernel",
            f"""
extern "C" __global__ void dense_matmul_kernel() {{
  float C_local[128];
  {WGMMA_CALL}
}}
""",
            {"block_M": 64, "block_N": 128, "block_K": 64},
            "dense_gemm",
            {"matmul", "gemm", "dense_gemm"},
            {"uses_gemm", "uses_wgmma"},
        ),
        (
            "mxfp4_matmul_kernel",
            f"""
extern "C" __global__ void mxfp4_matmul_kernel() {{
  float C_local[128];
  half_t B_dequantize_local[256];
  float scale_local[8];
  {WGMMA_CALL}
}}
""",
            {"block_M": 64, "block_N": 128, "block_K": 64},
            "blockscaled_gemm",
            {"matmul", "gemm", "quantized_gemm", "blockscaled_gemm"},
            {"uses_gemm", "has_dequant", "has_block_scale"},
        ),
        (
            "fp8_matmul_kernel",
            f"""
extern "C" __global__ void fp8_matmul_kernel() {{
  uint8_t A_local[64];
  uint8_t B_local[64];
  float C_local[128];
  {WGMMA_CALL}
}}
""",
            {"block_M": 64, "block_N": 128, "block_K": 64},
            "quantized_gemm",
            {"matmul", "gemm", "quantized_gemm"},
            {"uses_gemm", "has_quantized_dtype"},
        ),
        (
            "sparse_matmul_kernel",
            f"""
extern "C" __global__ void sparse_matmul_kernel(const void* BlockMask) {{
  float C_local[128];
  if (((bool*)BlockMask)[0]) {{}}
  {WGMMA_CALL}
}}
""",
            {"block_M": 64, "block_N": 128, "block_K": 64},
            "sparse_gemm",
            {"matmul", "gemm", "sparse_gemm"},
            {"uses_gemm", "has_sparse_mask"},
        ),
        (
            "flash_attention_kernel",
            f"""
extern "C" __global__ void flash_attention_kernel() {{
  float acc_s[128];
  float acc_o[64];
  float logsum[4];
  float scores_max[4];
  half_t acc_s_cast[128];
  {WGMMA_CALL.replace("C_local", "acc_s")}
}}
""",
            {"block_M": 64, "block_N": 128, "block_DK": 64},
            "flash_attention",
            {"attention", "flash_attention"},
            {"uses_gemm", "has_attention_state", "has_online_softmax"},
        ),
        (
            "flash_decode_kernel",
            """
extern "C" __global__ void flash_decode_kernel() {
  float acc_s[64];
  float acc_o[64];
  float scores_sum[4];
  int kv_seqlen = 0;
}
""",
            {"block_M": 16, "block_N": 64, "block_DK": 64},
            "flash_decoding",
            {"attention", "flash_decoding"},
            {"has_attention_state"},
        ),
        (
            "mha_sink_fwd_kernel",
            """
extern "C" __global__ void mha_sink_fwd_kernel(const void* Sinks) {
  float acc_s[128];
  float acc_o[64];
  float scores_max[4];
  (void)Sinks;
}
""",
            {"block_M": 64, "block_N": 128},
            "attention_sink",
            {"attention", "attention_sink"},
            {"has_attention_state"},
        ),
        (
            "block_sparse_attention_kernel",
            """
extern "C" __global__ void block_sparse_attention_kernel(const void* BlockIndices) {
  float acc_s[128];
  float acc_o[64];
  int selected_blocks = 4;
  (void)BlockIndices;
  (void)selected_blocks;
}
""",
            {"block_M": 64, "block_N": 128, "selected_blocks": 4},
            "block_sparse_attention",
            {"attention", "block_sparse_attention"},
            {"has_attention_state", "has_block_sparse_attention"},
        ),
        (
            "mla_decode_kernel",
            """
extern "C" __global__ void mla_decode_kernel() {
  float acc_s[128];
  float acc_o[64];
  int kv_lora_rank = 512;
  (void)kv_lora_rank;
}
""",
            {"block_M": 64, "block_N": 128},
            "mla",
            {"attention", "mla"},
            {"has_attention_state"},
        ),
        (
            "nsa_fwd_kernel",
            """
extern "C" __global__ void nsa_fwd_kernel(const void* BlockIndices) {
  float acc_s[128];
  float acc_o[64];
  int selected_blocks = 8;
  (void)BlockIndices;
  (void)selected_blocks;
}
""",
            {"block_M": 64, "block_N": 128, "selected_blocks": 8},
            "nsa",
            {"attention", "nsa"},
            {"has_attention_state", "has_block_sparse_attention"},
        ),
        (
            "conv_im2col_kernel",
            f"""
extern "C" __global__ void conv_im2col_kernel() {{
  float out_local[128];
  int im2col = 1;
  {WGMMA_CALL.replace("C_local", "out_local")}
  (void)im2col;
}}
""",
            {"block_M": 64, "block_N": 128, "block_K": 32},
            "conv",
            {"conv"},
            {"uses_gemm", "uses_im2col"},
        ),
        (
            "gemv_alloc_reducer_kernel",
            """
extern "C" __global__ void gemv_alloc_reducer_kernel() {
  float C_accum[1];
  float o_reducer[128];
  int reduce_threads = 32;
  // finalize_reducer is emitted by the tile op path.
  (void)reduce_threads;
}
""",
            {"BLOCK_N": 128, "reduce_threads": 32},
            "gemv",
            {"matmul", "gemv"},
            {"has_gemv"},
        ),
        (
            "tl_topk_kernel",
            """
extern "C" __global__ void tl_topk_kernel() {
  float topk_gates[8];
  int topk_indices[8];
  int max_idx = 0;
  reduce_max(topk_gates[0]);
  (void)topk_indices;
  (void)max_idx;
}
""",
            {"topk": 8},
            "topk",
            {"reduction", "topk"},
            {"has_topk", "uses_reduction"},
        ),
        (
            "chunk_delta_h_fwd_kernel",
            """
extern "C" __global__ void chunk_delta_h_fwd_kernel() {
  float hidden_shared[128];
  float da_cumsum[64];
  float prev_states[64];
  float dt_local[64];
  tl::gemm(hidden_shared, da_cumsum, prev_states);
}
""",
            {"block_T": 64, "block_DK": 64},
            "linear_attention",
            {"linear_attention"},
            {"uses_gemm", "has_linear_attention_state"},
        ),
    ],
)
def test_cuda_source_classifier_detects_kernel_families(
    function_name,
    source,
    config,
    expected_type,
    expected_tags,
    expected_traits,
):
    info, classification = _classify_source(function_name, source, config=config)

    assert info.detected_kernel_type == expected_type
    assert classification.primary_kernel_type == expected_type
    assert expected_tags.issubset(set(classification.kernel_type_tags))
    assert expected_traits.issubset(set(classification.traits))


def test_user_kernel_type_and_traits_are_reserved_override_points():
    info, classification = _classify_source(
        "dense_matmul_kernel",
        f"""
extern "C" __global__ void dense_matmul_kernel() {{
  float C_local[128];
  {WGMMA_CALL}
}}
""",
        config={"block_M": 64, "block_N": 128, "block_K": 64},
        filter_config=AutotuneFilterConfig(
            enabled=True,
            kernel_type="attention",
            kernel_traits=("user_supplied_trait",),
        ),
    )

    assert info.detected_kernel_type == "dense_gemm"
    assert classification.primary_kernel_type == "attention"
    assert classification.kernel_type_tags == ("attention",)
    assert "user_supplied_trait" in classification.traits


def test_tilelang_autotune_examples_have_family_expectations():
    expected_family_names = set(EXPECTED_AUTOTUNE_EXAMPLE_FAMILIES.values())
    assert expected_family_names.issubset(KNOWN_AUTOTUNE_FAMILIES)

    discovered_dirs = set()
    for path in (_REPO_ROOT / "examples").rglob("*.py"):
        rel_path = path.relative_to(_REPO_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        if not _AUTOTUNE_ENTRY_RE.search(source):
            continue
        parts = rel_path.split("/")
        discovered_dirs.add("/".join(parts[:2]))

    assert sorted(discovered_dirs - EXPECTED_AUTOTUNE_EXAMPLE_FAMILIES.keys()) == []
    assert sorted(EXPECTED_AUTOTUNE_EXAMPLE_FAMILIES.keys() - discovered_dirs) == []
