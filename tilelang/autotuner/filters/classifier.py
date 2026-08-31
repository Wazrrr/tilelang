"""Kernel-family classification helpers for autotune filter dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tilelang.autotuner.filters.common import KernelType

ATTENTION_SOFTMAX_NAMES = frozenset(
    {
        "logsum",
        "scores_max",
        "scores_max_prev",
        "scores_max_clear",
        "scores_scale",
        "scores_sum",
    }
)

PRE_COMPILE_WGMMA_OP_NAMES = frozenset(
    {
        "tl.ptx_wgmma_ss",
        "tl.ptx_wgmma_rs",
        "tl.ptx_wgmma_sp_ss",
        "tl.ptx_wgmma_sp_rs",
    }
)
PRE_COMPILE_MATMUL_OP_NAMES = frozenset(
    {
        "tl.tileop.gemm",
        "tl.gemm",
        "tl.ptx_mma_sm70",
        "tl.ptx_mma_block_scale",
        "tl.ptx_tcgen05_mma_ss",
        "tl.ptx_tcgen05_mma_ts",
        "tl.ptx_tcgen05_mma_blockscaled_ss",
        "tl.tvm_mfma",
        "tl.tvm_rdna_wmma",
    }
) | PRE_COMPILE_WGMMA_OP_NAMES
PRE_COMPILE_TMA_OP_NAMES = frozenset(
    {
        "tl.tma_load",
        "tl.tileop.tma_copy",
        "tl.tma_store_arrive",
        "tl.tma_store_wait",
        "tl.create_tma_descriptor",
    }
)
PRE_COMPILE_REDUCTION_OP_NAMES = frozenset(
    {
        "tl.tileop.finalize_reducer",
        "tl.warp_reduce_sum",
        "tl.warp_reduce_max",
        "tl.warp_reduce_min",
    }
)
PRE_COMPILE_ATOMIC_OP_PREFIXES = (
    "tl.atomic_",
    "tl.tileop.atomic",
)
PRE_COMPILE_SOFTMAX_OP_NAMES = frozenset(
    {
        "tl.__exp",
        "tl.__exp10",
        "tl.__log",
        "tl.__log2",
        "tl.__log10",
    }
)

_GEMM_SOURCE_TOKENS = (
    "tl::gemm",
    "wgmma",
    "mma_async",
    "tcgen05",
    "tvm_mfma",
    "mfma",
    "rdna_wmma",
    "wmma",
)
_QUANT_SOURCE_TOKENS = (
    "dequant",
    "fp4",
    "fp8",
    "float8",
    "uint8_t",
    "int8_t",
    "uint8",
    "int8",
    "e4m3",
    "e5m2",
    "mxfp",
    "nvfp",
    "uint4",
    "int4",
    "qweight",
    "q_weight",
)
_BLOCK_SCALE_SOURCE_TOKENS = (
    "blockscale",
    "block_scale",
    "blockscaled",
    "scale_local",
    "scale_shared",
    "scale_factor",
    "scale_b",
    "mxfp",
    "nvfp",
    "fp4",
)
_SPARSE_GEMM_SOURCE_TOKENS = (
    "blockmask",
    "block_mask",
    "sparse_mask",
)
_BLOCK_SPARSE_ATTENTION_TOKENS = (
    "blockindices",
    "block_indices",
    "selected_blocks",
    "selectedblock",
    "block_sparse",
    "sparse_attention",
)
_MLA_TOKENS = (
    "mla",
    "lora",
    "kv_lora",
    "qk_rope",
    "c_kv",
)
_NSA_TOKENS = (
    "nsa",
    "native_sparse_attention",
)
_FLASH_DECODING_TOKENS = (
    "decode",
    "decoding",
    "paged_kv",
    "kv_seqlen",
    "split_kv",
)
_ATTENTION_SINK_TOKENS = (
    "attention_sink",
    "sink",
    "sinks",
)
_FLASH_ATTENTION_TOKENS = (
    "flash",
    "attn",
    "mha",
    "gqa",
)
_TOPK_TOKENS = (
    "topk",
    "top_k",
    "topk_indices",
    "topk_gates",
    "max_idx",
)
_LINEAR_ATTENTION_STRONG_TOKENS = (
    "linear_attention",
    "mamba",
    "chunk",
    "chunk_scan",
    "chunk_state",
    "kda",
    "gdn",
    "gla",
    "wy_fast",
    "delta_h",
    "delta_rule",
    "prev_state",
    "prev_states",
    "da_cumsum",
    "cumsum",
)
_LINEAR_ATTENTION_CONTEXT_TOKENS = (
    "hidden",
    "da_",
    "dt",
)
_GEMV_TOKENS = (
    "gemv",
    "alloc_reducer",
    "finalize_reducer",
    "o_reducer",
    "c_accum",
    "reduce_threads",
)
_IM2COL_TOKENS = (
    "im2col",
    "tileop.im2col",
)


@dataclass(frozen=True)
class KernelClassification:
    """Resolved kernel type hierarchy and traits used for rule dispatch."""

    primary_kernel_type: str = "generic"
    kernel_type_tags: tuple[str, ...] = ("generic",)
    traits: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_kernel_type": self.primary_kernel_type,
            "kernel_type_tags": list(self.kernel_type_tags),
            "traits": list(self.traits),
            "evidence": list(self.evidence),
        }


def normalize_kernel_names(values: Any) -> tuple[str, ...]:
    """Normalize one user-provided kernel name or list of names."""
    if values is None:
        return ()
    if isinstance(values, str):
        values = (values,)
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        name = str(value).strip()
        if not name or name in seen:
            continue
        normalized.append(name)
        seen.add(name)
    return tuple(normalized)


def merge_kernel_names(*groups: tuple[str, ...]) -> tuple[str, ...]:
    """Merge kernel type tags, traits, or evidence while preserving order."""
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for name in group:
            if name and name not in seen:
                merged.append(name)
                seen.add(name)
    return tuple(merged)


def classify_kernel_filter_info(info: Any, config: Any) -> KernelClassification:
    """Resolve user-provided kernel hints against automatic classifier output."""
    detected_type = str(getattr(info, "detected_kernel_type", None) or "generic")
    detected_traits = tuple(getattr(info, "detected_kernel_traits", ()) or ())
    explicit_traits = normalize_kernel_names(getattr(config, "kernel_traits", ()))
    evidence = list(getattr(info, "classification_evidence", ()) or ())
    explicit_kernel_type = str(getattr(config, "kernel_type", "auto") or "auto")

    if explicit_kernel_type != "auto":
        primary_kernel_type = explicit_kernel_type
        evidence.append(f"user.kernel_type={primary_kernel_type}")
    else:
        primary_kernel_type = "generic" if detected_type == "auto" else detected_type

    traits = merge_kernel_names(detected_traits, explicit_traits)
    if explicit_traits:
        evidence.append(f"user.kernel_traits={','.join(explicit_traits)}")
    kernel_type_tags = expand_kernel_type_tags(primary_kernel_type, traits)
    return KernelClassification(
        primary_kernel_type=primary_kernel_type,
        kernel_type_tags=kernel_type_tags,
        traits=traits,
        evidence=tuple(evidence),
    )


def expand_kernel_type_tags(
    kernel_type: str,
    traits: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Expand a primary kernel type into the tags consumed by rules."""
    mapping: dict[str, tuple[str, ...]] = {
        "generic": ("generic",),
        "matmul": ("matmul",),
        "gemm": ("matmul", "gemm"),
        "dense_gemm": ("matmul", "gemm", "dense_gemm"),
        "grouped_gemm": ("matmul", "gemm", "grouped_gemm"),
        "splitk_gemm": ("matmul", "gemm", "splitk_gemm"),
        "streamk_gemm": ("matmul", "gemm", "streamk_gemm"),
        "quantized_gemm": ("matmul", "gemm", "quantized_gemm"),
        "blockscaled_gemm": ("matmul", "gemm", "quantized_gemm", "blockscaled_gemm"),
        "sparse_gemm": ("matmul", "gemm", "sparse_gemm"),
        "gemv": ("matmul", "gemv"),
        "attention": ("attention",),
        "flash_attention": ("attention", "flash_attention"),
        "flash_decoding": ("attention", "flash_decoding"),
        "block_sparse_attention": ("attention", "block_sparse_attention"),
        "mla": ("attention", "mla"),
        "nsa": ("attention", "nsa"),
        "attention_sink": ("attention", "attention_sink"),
        "reduction": ("reduction",),
        "softmax": ("reduction", "softmax"),
        "norm": ("reduction", "norm"),
        "topk": ("reduction", "topk"),
        "scan": ("scan",),
        "linear_attention": ("linear_attention",),
        "conv": ("conv",),
        "elementwise": ("elementwise",),
        "cast": ("elementwise", "cast"),
    }
    expanded_types = mapping.get(kernel_type)
    if expanded_types is not None:
        return expanded_types
    if "uses_gemm" in traits:
        return ("matmul", kernel_type)
    return (kernel_type,)


def detect_cuda_kernel_traits(
    *,
    function_name: str,
    c_local_matches: list[int],
    fragment_array_elements: dict[str, int],
    source: str,
    wgmma_shapes: list[tuple[int, int, int]],
    tma_load_count: int,
    tma_store_count: int,
    stmatrix_count: int,
    mbarrier_count: int,
    syncthreads_count: int,
    sparse_mask_access_count: int,
    config: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Detect kernel traits from emitted CUDA source and autotune config keys."""
    traits: list[str] = []
    evidence: list[str] = []
    source_text = _classification_text(function_name, fragment_array_elements, source, config)
    source_lower = source.lower()

    def add(trait: str, reason: str) -> None:
        if trait not in traits:
            traits.append(trait)
        evidence.append(reason)

    if wgmma_shapes:
        add("uses_wgmma", f"source.wgmma_shapes={len(wgmma_shapes)}")
        add("uses_gemm", "source.uses_wgmma")
    if _has_any(source_lower, _GEMM_SOURCE_TOKENS):
        add("uses_gemm", "source.matmul_signal")
    if c_local_matches:
        add("uses_gemm", f"source.c_local_arrays={len(c_local_matches)}")
        add("has_accumulator_tile", f"source.max_c_local={max(c_local_matches)}")
    if any(is_dequant_fragment_name(name) for name in fragment_array_elements) or _has_any(source_text, _QUANT_SOURCE_TOKENS):
        add("has_dequant", "source.quant_or_dequant_signal")
    if _has_any(source_text, _BLOCK_SCALE_SOURCE_TOKENS):
        add("has_block_scale", "source.block_scale_signal")
    if _has_any(source_text, ("fp8", "float8", "uint8_t", "int8_t", "uint8", "int8", "e4m3", "e5m2")):
        add("has_quantized_dtype", "source.quantized_dtype_signal")
    if sparse_mask_access_count:
        add("has_sparse_mask", f"source.sparse_mask_accesses={sparse_mask_access_count}")
    if _has_any(source_text, _BLOCK_SPARSE_ATTENTION_TOKENS):
        add("has_block_sparse_attention", "source.block_sparse_attention_signal")
    if "acc_s" in fragment_array_elements and "acc_o" in fragment_array_elements:
        add("uses_gemm", "source.attention_accumulators")
        add("has_attention_state", "source.acc_s_and_acc_o")
        add("has_softmax", "source.attention_state")
    if any(name in fragment_array_elements for name in ATTENTION_SOFTMAX_NAMES):
        add("has_online_softmax", "source.softmax_state_fragments")
    if "exp2(" in source or "expf(" in source or "exp(" in source:
        add("has_exp", "source.exp")
    if "reduce_max" in source or "reduce_sum" in source:
        add("uses_reduction", "source.reduction")
    if _has_any(source_text, _TOPK_TOKENS) or _config_has_any_key(config, ("topk", "top_k")):
        add("has_topk", "source_or_config.topk_signal")
    if _has_linear_attention_signal(source_text):
        add("has_linear_attention_state", "source.linear_attention_signal")
    if _has_any(source_text, ("scan", "chunk_scan", "cumsum")):
        add("has_scan", "source.scan_signal")
    if _has_any(source_text, _GEMV_TOKENS):
        add("has_gemv", "source.gemv_signal")
    if _has_any(source_text, _IM2COL_TOKENS):
        add("uses_im2col", "source.im2col")
    if tma_load_count or tma_store_count:
        add("uses_tma", f"source.tma_loads={tma_load_count},stores={tma_store_count}")
    if stmatrix_count:
        add("uses_stmatrix", f"source.stmatrix={stmatrix_count}")
    if mbarrier_count:
        add("uses_mbarrier", f"source.mbarrier={mbarrier_count}")
    if syncthreads_count:
        add("uses_syncthreads", f"source.syncthreads={syncthreads_count}")
    if _has_tiled_mnk_config(config):
        add("has_tiled_mnk_config", "config.block_M/block_N/block_K")

    return tuple(traits), tuple(evidence)


def detect_pre_compile_kernel_traits(
    *,
    wgmma_shapes: list[tuple[int, int, int]],
    op_names: frozenset[str],
    config: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Detect kernel traits from lowered device IR before CUDA compilation."""
    traits: list[str] = []
    evidence: list[str] = []
    normalized_ops = frozenset(str(op_name).lower() for op_name in op_names)

    def add(trait: str, reason: str) -> None:
        if trait not in traits:
            traits.append(trait)
        evidence.append(reason)

    if wgmma_shapes:
        add("uses_wgmma", f"ir.wgmma_shapes={len(wgmma_shapes)}")
        add("uses_gemm", "ir.uses_wgmma")
    if normalized_ops & PRE_COMPILE_MATMUL_OP_NAMES:
        add("uses_gemm", "ir.matmul_ops")
    if normalized_ops & PRE_COMPILE_TMA_OP_NAMES:
        add("uses_tma", "ir.tma_ops")
    if normalized_ops & PRE_COMPILE_REDUCTION_OP_NAMES:
        add("uses_reduction", "ir.reduction_ops")
    if normalized_ops & PRE_COMPILE_SOFTMAX_OP_NAMES:
        add("has_exp_or_log", "ir.exp_or_log_ops")
    if any(op_name.startswith(PRE_COMPILE_ATOMIC_OP_PREFIXES) for op_name in normalized_ops):
        add("uses_atomic", "ir.atomic_ops")
    if any("im2col" in op_name for op_name in normalized_ops):
        add("uses_im2col", "ir.im2col_ops")
    if any("topk" in op_name or "top_k" in op_name for op_name in normalized_ops):
        add("has_topk", "ir.topk_ops")
    if any("scan" in op_name for op_name in normalized_ops):
        add("has_scan", "ir.scan_ops")
    if "tl.tileop.alloc_reducer" in normalized_ops:
        add("has_gemv", "ir.alloc_reducer")
    if _has_tiled_mnk_config(config):
        add("has_tiled_mnk_config", "config.block_M/block_N/block_K")

    return tuple(traits), tuple(evidence)


def detect_pre_compile_kernel_type(traits: tuple[str, ...]) -> KernelType:
    """Choose a conservative family from IR-only traits."""
    trait_set = set(traits)
    if "uses_im2col" in trait_set:
        return "conv"
    if "has_topk" in trait_set:
        return "topk"
    if "has_linear_attention_state" in trait_set:
        return "linear_attention"
    if "has_gemv" in trait_set and "uses_gemm" not in trait_set:
        return "gemv"
    if "uses_gemm" in trait_set:
        return "generic"
    if "uses_reduction" in trait_set and "has_exp_or_log" in trait_set:
        return "softmax"
    if "uses_reduction" in trait_set:
        return "reduction"
    if "has_scan" in trait_set:
        return "scan"
    return "generic"


def detect_cuda_kernel_type(
    *,
    c_local_matches: list[int],
    fragment_array_elements: dict[str, int],
    source: str,
    function_name: str,
    traits: tuple[str, ...] = (),
    sparse_mask_access_count: int = 0,
    config: dict[str, Any] | None = None,
) -> KernelType:
    """Choose the primary kernel family from source/config traits."""
    trait_set = set(traits)
    config = config or {}
    text = _classification_text(function_name, fragment_array_elements, source, config)

    if "uses_im2col" in trait_set or _has_any(text, _IM2COL_TOKENS):
        return "conv"

    if "has_attention_state" in trait_set:
        if _has_any(text, _MLA_TOKENS):
            return "mla"
        if _has_any(text, _NSA_TOKENS):
            return "nsa"
        if _has_any(text, _ATTENTION_SINK_TOKENS):
            return "attention_sink"
        if "has_block_sparse_attention" in trait_set or _has_any(text, _BLOCK_SPARSE_ATTENTION_TOKENS):
            return "block_sparse_attention"
        if _has_any(text, _FLASH_DECODING_TOKENS):
            return "flash_decoding"
        if _has_any(text, _FLASH_ATTENTION_TOKENS):
            return "flash_attention"
        return "attention"

    if "has_topk" in trait_set:
        return "topk"
    if "has_linear_attention_state" in trait_set:
        return "linear_attention"
    if "has_scan" in trait_set and "uses_gemm" not in trait_set:
        return "scan"
    if sparse_mask_access_count > 0 or "has_sparse_mask" in trait_set:
        return "sparse_gemm" if "uses_gemm" in trait_set else "generic"
    if "has_block_scale" in trait_set and "uses_gemm" in trait_set:
        return "blockscaled_gemm"
    if ("has_dequant" in trait_set or "has_quantized_dtype" in trait_set) and "uses_gemm" in trait_set:
        return "quantized_gemm"
    if "has_gemv" in trait_set:
        return "gemv"
    if c_local_matches:
        return "dense_gemm"
    if "uses_gemm" in trait_set and "has_tiled_mnk_config" in trait_set:
        return "dense_gemm"
    if "uses_reduction" in trait_set and ("has_exp_or_log" in trait_set or "has_exp" in trait_set or "has_softmax" in trait_set):
        return "softmax"
    if "uses_reduction" in trait_set:
        return "reduction"
    return "generic"


def is_dequant_fragment_name(name: str) -> bool:
    return "dequant" in name.lower()


def _classification_text(
    function_name: str,
    fragment_array_elements: dict[str, int],
    source: str,
    config: dict[str, Any],
) -> str:
    config_keys = " ".join(str(key) for key in config)
    fragment_names = " ".join(fragment_array_elements)
    return f"{function_name} {fragment_names} {config_keys} {source}".lower()


def _has_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token.lower() in text for token in tokens)


def _has_linear_attention_signal(text: str) -> bool:
    if _has_any(text, _LINEAR_ATTENTION_STRONG_TOKENS):
        return True
    context_hits = sum(1 for token in _LINEAR_ATTENTION_CONTEXT_TOKENS if token.lower() in text)
    return context_hits >= 2


def _config_has_any_key(config: dict[str, Any], keys: tuple[str, ...]) -> bool:
    normalized = {str(key).lower() for key in config}
    return any(key.lower() in normalized for key in keys)


def _has_tiled_mnk_config(config: dict[str, Any]) -> bool:
    return (
        _config_int(config, "block_M") is not None
        and _config_int(config, "block_N") is not None
        and _config_int(config, "block_K") is not None
    )


def _config_int(config: dict[str, Any], key: str) -> int | None:
    for candidate in (key, key.lower(), key.upper()):
        value = config.get(candidate)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


__all__ = [
    "ATTENTION_SOFTMAX_NAMES",
    "PRE_COMPILE_ATOMIC_OP_PREFIXES",
    "PRE_COMPILE_MATMUL_OP_NAMES",
    "PRE_COMPILE_REDUCTION_OP_NAMES",
    "PRE_COMPILE_SOFTMAX_OP_NAMES",
    "PRE_COMPILE_TMA_OP_NAMES",
    "PRE_COMPILE_WGMMA_OP_NAMES",
    "KernelClassification",
    "classify_kernel_filter_info",
    "detect_cuda_kernel_traits",
    "detect_cuda_kernel_type",
    "detect_pre_compile_kernel_traits",
    "detect_pre_compile_kernel_type",
    "expand_kernel_type_tags",
    "is_dequant_fragment_name",
    "merge_kernel_names",
    "normalize_kernel_names",
]
