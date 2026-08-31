"""Common autotune filters shared by most CUDA kernels."""

from __future__ import annotations

from tilelang.autotuner.filters.rule_sets.base import (
    ATTENTION_KERNEL_TYPE_TAGS,
    POST_COMPILE_STAGES,
    LimitFilterRule,
)
from tilelang.autotuner.filters.rules import AutotuneVerifyRule


def make_common_filter_rules() -> list[AutotuneVerifyRule]:
    """Return resource-pressure filters that apply across kernel families."""
    return [
        LimitFilterRule(
            name="common.spills",
            layer="common",
            finding_kind="violation",
            info_attr="n_spills",
            enabled_attr="check_spills",
            limit_attr="max_spills",
            reason="spills_over_limit",
            exclude_kernel_type_tags=ATTENTION_KERNEL_TYPE_TAGS,
            stages=POST_COMPILE_STAGES,
        ),
        LimitFilterRule(
            name="common.local_memory",
            layer="common",
            finding_kind="violation",
            info_attr="local_size_bytes",
            enabled_attr="check_local_memory",
            limit_attr="max_local_size_bytes",
            reason="local_memory_over_limit",
            exclude_kernel_type_tags=ATTENTION_KERNEL_TYPE_TAGS,
            stages=POST_COMPILE_STAGES,
        ),
        LimitFilterRule(
            name="common.registers",
            layer="common",
            finding_kind="violation",
            info_attr="n_regs",
            enabled_attr="check_registers",
            limit_attr="max_registers_per_thread",
            reason="registers_per_thread_over_limit",
            stages=POST_COMPILE_STAGES,
        ),
    ]
