"""Kernel-family-specific autotune filters."""

from __future__ import annotations

from typing import Any

from tilelang.autotuner.filters.rule_sets.base import (
    ATTENTION_KERNEL_TYPE_TAGS,
    POST_COMPILE_STAGES,
    PRE_COMPILE_STAGES,
    QUANTIZED_GEMM_KERNEL_TYPE_TAGS,
    SPARSE_GEMM_KERNEL_TYPE_TAGS,
    LimitFilterRule,
)
from tilelang.autotuner.filters.rules import AutotuneRuleContext, AutotuneVerifyRule, RuleFindingKind, RuleLayer


class SparseMaskRequiredRule(AutotuneVerifyRule):
    """Optionally reject sparse GEMM candidates that do not use the block mask."""

    name = "sparse_gemm.sparse_mask_required"
    layer: RuleLayer = "kernel"
    finding_kind: RuleFindingKind = "violation"
    match_kernel_type_tags = SPARSE_GEMM_KERNEL_TYPE_TAGS
    stages = PRE_COMPILE_STAGES

    def check(self, context: AutotuneRuleContext) -> list[dict[str, Any]]:
        config = context.config
        info = context.info
        if config.check_sparse_mask and info.source_available and info.sparse_mask_access_count == 0:
            return [
                {
                    "reason": "sparse_mask_not_detected",
                    "function": info.function_name,
                    "observed": info.sparse_mask_access_count,
                }
            ]
        return []


def make_attention_post_compile_filter_rules() -> list[AutotuneVerifyRule]:
    """Return attention-family hard filters that use PTXAS resource usage."""
    return [
        LimitFilterRule(
            name="attention.spills",
            layer="kernel",
            finding_kind="violation",
            info_attr="n_spills",
            enabled_attr="check_attention_spills",
            limit_attr="max_attention_spills",
            reason="attention_spills_over_limit",
            match_kernel_type_tags=ATTENTION_KERNEL_TYPE_TAGS,
            stages=POST_COMPILE_STAGES,
        ),
        LimitFilterRule(
            name="attention.local_memory",
            layer="kernel",
            finding_kind="violation",
            info_attr="local_size_bytes",
            enabled_attr="check_attention_local_memory",
            limit_attr="max_attention_local_size_bytes",
            reason="attention_local_memory_over_limit",
            match_kernel_type_tags=ATTENTION_KERNEL_TYPE_TAGS,
            stages=POST_COMPILE_STAGES,
        ),
    ]


def make_attention_pre_compile_filter_rules() -> list[AutotuneVerifyRule]:
    """Return attention-family filters that use IR/CUDA-source features."""
    return [
        LimitFilterRule(
            name="attention.state_elements",
            layer="kernel",
            finding_kind="advisory",
            info_attr="attention_state_elements_per_thread",
            enabled_attr="check_attention_state_elements_per_thread",
            limit_attr="max_attention_state_elements_per_thread",
            reason="attention_state_elements_per_thread_over_limit",
            match_kernel_type_tags=ATTENTION_KERNEL_TYPE_TAGS,
            stages=PRE_COMPILE_STAGES,
        ),
    ]


def make_gemm_filter_rules() -> list[AutotuneVerifyRule]:
    """Return GEMM-family filters."""
    return [
        LimitFilterRule(
            name="gemm.c_local",
            layer="kernel",
            finding_kind="advisory",
            info_attr="c_local_floats",
            enabled_attr="check_c_local",
            limit_attr="max_c_local_floats",
            reason="c_local_floats_over_limit",
            exclude_kernel_type_tags=ATTENTION_KERNEL_TYPE_TAGS,
            exclude_kernel_traits=frozenset({"uses_wgmma"}),
            stages=PRE_COMPILE_STAGES,
        ),
    ]


def make_quantized_gemm_filter_rules() -> list[AutotuneVerifyRule]:
    """Return quantized and block-scaled GEMM filters."""
    return [
        LimitFilterRule(
            name="quantized_gemm.dequant_elements",
            layer="kernel",
            finding_kind="advisory",
            info_attr="quant_dequant_elements_per_thread",
            enabled_attr="check_quant_dequant_elements_per_thread",
            limit_attr="max_quant_dequant_elements_per_thread",
            reason="quant_dequant_elements_per_thread_over_limit",
            match_kernel_type_tags=QUANTIZED_GEMM_KERNEL_TYPE_TAGS,
            stages=PRE_COMPILE_STAGES,
        ),
    ]


def make_sparse_gemm_filter_rules() -> list[AutotuneVerifyRule]:
    """Return sparse GEMM filters."""
    return [SparseMaskRequiredRule()]
