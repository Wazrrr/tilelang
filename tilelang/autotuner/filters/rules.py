"""Rule registry used by autotune candidate filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RuleLayer = Literal["common", "primitive", "kernel"]
RuleFindingKind = Literal["violation", "advisory", "observation"]


@dataclass(frozen=True)
class AutotuneRuleContext:
    """Runtime inputs passed to one autotune verification rule."""

    info: Any
    config: Any
    stage: str
    primary_kernel_type: str
    kernel_type_tags: frozenset[str] = field(default_factory=lambda: frozenset({"generic"}))
    kernel_traits: frozenset[str] = field(default_factory=frozenset)
    classification_evidence: tuple[str, ...] = ()


class AutotuneVerifyRule:
    """Base class for pluggable autotune verification rules."""

    name: str = ""
    layer: RuleLayer = "common"
    finding_kind: RuleFindingKind = "violation"
    match_kernel_type_tags: frozenset[str] | None = None
    exclude_kernel_type_tags: frozenset[str] = frozenset()
    required_kernel_traits: frozenset[str] | None = None
    exclude_kernel_traits: frozenset[str] = frozenset()
    stages: frozenset[str] | None = None

    def applies_to(self, context: AutotuneRuleContext) -> bool:
        if self.stages is not None and context.stage not in self.stages:
            return False
        if self.match_kernel_type_tags is not None and not (self.match_kernel_type_tags & context.kernel_type_tags):
            return False
        if self.exclude_kernel_type_tags & context.kernel_type_tags:
            return False
        if self.required_kernel_traits is not None and not self.required_kernel_traits.issubset(context.kernel_traits):
            return False
        return not (self.exclude_kernel_traits & context.kernel_traits)

    def check(self, context: AutotuneRuleContext) -> list[dict[str, Any]]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, layer={self.layer!r}, finding_kind={self.finding_kind!r})"


class AutotuneRuleRegistry:
    """Ordered plugin registry for autotune verification rules."""

    def __init__(self, rules: list[AutotuneVerifyRule] | None = None):
        self._rules: list[AutotuneVerifyRule] = []
        for rule in rules or []:
            self.register(rule)

    def register(self, rule: AutotuneVerifyRule, *, replace: bool = False) -> AutotuneVerifyRule:
        if not rule.name:
            raise ValueError(f"{type(rule).__name__} must define a non-empty rule name")
        existing_index = self._find_index(rule.name)
        if existing_index is not None:
            if not replace:
                raise ValueError(f"Autotune verify rule '{rule.name}' is already registered")
            self._rules[existing_index] = rule
            return rule
        self._rules.append(rule)
        return rule

    def unregister(self, name: str) -> AutotuneVerifyRule | None:
        existing_index = self._find_index(name)
        if existing_index is None:
            return None
        return self._rules.pop(existing_index)

    def clear(self) -> None:
        self._rules.clear()

    def copy(self) -> AutotuneRuleRegistry:
        return AutotuneRuleRegistry(list(self._rules))

    def rules(
        self,
        *,
        layer: RuleLayer | None = None,
        finding_kind: RuleFindingKind | None = None,
    ) -> list[AutotuneVerifyRule]:
        rules = self._rules
        if layer is not None:
            rules = [rule for rule in rules if rule.layer == layer]
        if finding_kind is not None:
            rules = [rule for rule in rules if rule.finding_kind == finding_kind]
        return list(rules)

    def rules_for(
        self,
        context: AutotuneRuleContext,
        *,
        finding_kind: RuleFindingKind,
    ) -> list[AutotuneVerifyRule]:
        return [rule for rule in self._rules if rule.finding_kind == finding_kind and rule.applies_to(context)]

    def _find_index(self, name: str) -> int | None:
        for index, rule in enumerate(self._rules):
            if rule.name == name:
                return index
        return None
