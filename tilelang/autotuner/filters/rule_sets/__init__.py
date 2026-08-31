"""Built-in autotune filter rule sets."""

from __future__ import annotations

from tilelang.autotuner.filters.rule_sets.common import make_common_filter_rules
from tilelang.autotuner.filters.rule_sets.kernel_family import (
    make_attention_post_compile_filter_rules,
    make_attention_pre_compile_filter_rules,
    make_gemm_filter_rules,
    make_quantized_gemm_filter_rules,
    make_sparse_gemm_filter_rules,
)
from tilelang.autotuner.filters.rule_sets.primitive import make_primitive_filter_rules
from tilelang.autotuner.filters.rules import AutotuneVerifyRule


def make_default_filter_rules() -> list[AutotuneVerifyRule]:
    """Return built-in rules in the same evaluation order as the old registry."""
    return [
        *make_attention_post_compile_filter_rules(),
        *make_common_filter_rules(),
        *make_attention_pre_compile_filter_rules(),
        *make_gemm_filter_rules(),
        *make_primitive_filter_rules(),
        *make_quantized_gemm_filter_rules(),
        *make_sparse_gemm_filter_rules(),
    ]


__all__ = [
    "make_attention_post_compile_filter_rules",
    "make_attention_pre_compile_filter_rules",
    "make_common_filter_rules",
    "make_default_filter_rules",
    "make_gemm_filter_rules",
    "make_primitive_filter_rules",
    "make_quantized_gemm_filter_rules",
    "make_sparse_gemm_filter_rules",
]
