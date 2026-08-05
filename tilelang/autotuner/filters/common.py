"""Shared interfaces for autotune candidate filters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

FilterStage = Literal["pre_compile", "post_compile", "post_compile_quality"]
FilterVerdict = Literal["keep", "reject"]
FilterAction = Literal["reject", "report"]
KernelType = Literal["auto", "generic", "gemm", "attention"]


class AutotuneBaseFilterConfig:
    """Base config shared by hard resource and quality filters."""

    @classmethod
    def from_value(cls, value: bool | dict[str, Any] | "AutotuneBaseFilterConfig" | None):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls(enabled=value)
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(f"Unsupported {cls.__name__} config: {type(value)!r}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["report_path"] is not None:
            data["report_path"] = str(data["report_path"])
        return data

    def to_cache_key_dict(self) -> dict[str, Any]:
        data = self.to_dict()
        data.pop("report_path", None)
        return data


@dataclass(frozen=True)
class AutotuneFilterDecision:
    verdict: FilterVerdict
    stage: FilterStage
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def keep(self) -> bool:
        return self.verdict == "keep"

    @classmethod
    def keep_decision(cls, stage: FilterStage, reason: str, **details: Any) -> "AutotuneFilterDecision":
        return cls("keep", stage, reason, details)

    @classmethod
    def reject_decision(cls, stage: FilterStage, reason: str, **details: Any) -> "AutotuneFilterDecision":
        return cls("reject", stage, reason, details)
