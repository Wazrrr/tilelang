"""Primitive-specific autotune filters for WGMMA, TMA, and tile shape choices."""

from __future__ import annotations

from typing import Any

from tilelang.autotuner.filters.rule_sets.base import (
    ATTENTION_KERNEL_TYPE_TAGS,
    PRE_COMPILE_STAGES,
    LimitFilterRule,
)
from tilelang.autotuner.filters.rules import (
    AutotuneRuleContext,
    AutotuneVerifyRule,
    RuleFindingKind,
    RuleLayer,
)


class TmaTinyTileRule(AutotuneVerifyRule):
    """Flag TMA loads on small tiles with shallow software pipelines."""

    name = "primitive.tma_tiny_tile"
    layer: RuleLayer = "primitive"
    finding_kind: RuleFindingKind = "advisory"
    stages = PRE_COMPILE_STAGES

    def check(self, context: AutotuneRuleContext) -> list[dict[str, Any]]:
        config = context.config
        info = context.info
        if not (
            config.check_tma_tiny_tile
            and config.max_tma_tiny_tile_area is not None
            and info.tile_area is not None
            and info.tma_load_count > 0
            and info.tile_area <= config.max_tma_tiny_tile_area
            and (config.tma_tiny_tile_num_stages is None or info.num_stages == config.tma_tiny_tile_num_stages)
        ):
            return []
        return [
            {
                "reason": "tma_tiny_tile_limit",
                "function": info.function_name,
                "tile_area": info.tile_area,
                "limit": config.max_tma_tiny_tile_area,
                "num_stages": info.num_stages,
                "tma_load_count": info.tma_load_count,
            }
        ]


class WgmmaRegisterPressureObservation(AutotuneVerifyRule):
    """Record bounded accumulator pressure without changing the filter verdict."""

    name = "primitive.wgmma_register_pressure"
    layer: RuleLayer = "primitive"
    finding_kind: RuleFindingKind = "observation"
    stages = PRE_COMPILE_STAGES

    def check(self, context: AutotuneRuleContext) -> list[dict[str, Any]]:
        pressure = context.info.wgmma_register_pressure
        if not context.config.check_wgmma_register_pressure or pressure is None:
            return []
        return [
            {
                "reason": "wgmma_register_pressure_observed",
                "function": context.info.function_name,
                **pressure.to_dict(),
            }
        ]


class WgmmaRegisterPressureOverBudgetRule(AutotuneVerifyRule):
    """Flag WGMMA candidates whose live accumulators exceed their warp budget."""

    name = "primitive.wgmma_register_pressure_over_budget"
    layer: RuleLayer = "primitive"
    finding_kind: RuleFindingKind = "advisory"
    required_kernel_traits = frozenset({"uses_wgmma"})
    stages = PRE_COMPILE_STAGES

    def check(self, context: AutotuneRuleContext) -> list[dict[str, Any]]:
        pressure = context.info.wgmma_register_pressure
        if not context.config.check_wgmma_register_pressure or pressure is None or pressure.status != "over_budget":
            return []
        return [
            {
                "reason": "wgmma_register_pressure_over_budget",
                "function": context.info.function_name,
                "observed": pressure.lower_bound_registers,
                "limit": pressure.register_budget,
                **pressure.to_dict(),
            }
        ]


def make_primitive_filter_rules() -> list[AutotuneVerifyRule]:
    """Return primitive-level filters that are independent of one kernel family."""
    return [
        WgmmaRegisterPressureOverBudgetRule(),
        WgmmaRegisterPressureObservation(),
        LimitFilterRule(
            name="primitive.output_elements_per_thread",
            layer="primitive",
            finding_kind="advisory",
            info_attr="output_elements_per_thread",
            enabled_attr="check_output_elements_per_thread",
            limit_attr="max_output_elements_per_thread",
            reason="output_elements_per_thread_over_limit",
            exclude_kernel_type_tags=ATTENTION_KERNEL_TYPE_TAGS,
            exclude_kernel_traits=frozenset({"uses_wgmma"}),
            stages=PRE_COMPILE_STAGES,
        ),
        LimitFilterRule(
            name="primitive.k_loop",
            layer="primitive",
            finding_kind="advisory",
            info_attr="max_k_loop_iterations",
            enabled_attr="check_k_loop",
            limit_attr="max_k_loop_iterations",
            reason="k_loop_iterations_over_limit",
            stages=PRE_COMPILE_STAGES,
        ),
        LimitFilterRule(
            name="primitive.tma_store_count",
            layer="primitive",
            finding_kind="advisory",
            info_attr="tma_store_count",
            enabled_attr="check_tma_store_count",
            limit_attr="max_tma_store_count",
            reason="tma_store_count_over_limit",
            exclude_kernel_type_tags=ATTENTION_KERNEL_TYPE_TAGS,
            stages=PRE_COMPILE_STAGES,
        ),
        TmaTinyTileRule(),
    ]
