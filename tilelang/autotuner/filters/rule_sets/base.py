"""Shared building blocks for built-in autotune filter rules."""

from __future__ import annotations

from typing import Any

from tilelang.autotuner.filters.rules import (
    AutotuneRuleContext,
    AutotuneVerifyRule,
    RuleFindingKind,
    RuleLayer,
)

ATTENTION_KERNEL_TYPE_TAGS = frozenset({"attention"})
QUANTIZED_GEMM_KERNEL_TYPE_TAGS = frozenset({"quantized_gemm"})
SPARSE_GEMM_KERNEL_TYPE_TAGS = frozenset({"sparse_gemm"})

PRE_COMPILE_STAGES = frozenset({"pre_compile"})
POST_COMPILE_STAGES = frozenset({"post_compile"})


class LimitFilterRule(AutotuneVerifyRule):
    """Rule for checking one extracted numeric metric against one config limit."""

    def __init__(
        self,
        *,
        name: str,
        layer: RuleLayer,
        finding_kind: RuleFindingKind,
        info_attr: str,
        enabled_attr: str,
        limit_attr: str,
        reason: str,
        match_kernel_type_tags: frozenset[str] | None = None,
        exclude_kernel_type_tags: frozenset[str] = frozenset(),
        stages: frozenset[str] | None = None,
    ):
        self.name = name
        self.layer = layer
        self.finding_kind = finding_kind
        self.info_attr = info_attr
        self.enabled_attr = enabled_attr
        self.limit_attr = limit_attr
        self.reason = reason
        self.match_kernel_type_tags = match_kernel_type_tags
        self.exclude_kernel_type_tags = exclude_kernel_type_tags
        self.stages = stages

    def check(self, context: AutotuneRuleContext) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        append_limit_finding(
            findings,
            enabled=bool(getattr(context.config, self.enabled_attr)),
            observed=getattr(context.info, self.info_attr),
            limit=getattr(context.config, self.limit_attr),
            reason=self.reason,
            function=context.info.function_name,
        )
        return findings


def append_limit_finding(
    findings: list[dict[str, Any]],
    *,
    enabled: bool,
    observed: int | None,
    limit: int | None,
    reason: str,
    function: str,
) -> None:
    if not enabled or observed is None or limit is None:
        return
    if observed > limit:
        findings.append(
            {
                "reason": reason,
                "function": function,
                "observed": observed,
                "limit": limit,
            }
        )
