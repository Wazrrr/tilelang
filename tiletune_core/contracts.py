"""Versioned, JSON-only compiler/evaluator boundary.

Compiler adapters must resolve expressions and prove dependencies before export.
The core deliberately has no expression parser, simplifier, or ownership inference.
"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from math import isfinite

FACT_VERSION = 1
REPORT_VERSION = 1


def _check_json(item):
    if item is None or type(item) in (bool, str, int):
        return
    if type(item) is float and isfinite(item):
        return
    if type(item) in (list, tuple):
        for child in item:
            _check_json(child)
        return
    if type(item) is dict and all(type(key) is str for key in item):
        for child in item.values():
            _check_json(child)
        return
    raise ValueError(f"facts require finite JSON values, got {type(item).__name__}")


def json_value(value):
    """Copy a strict JSON value; never stringify an unresolved compiler object."""
    _check_json(value)
    return json.loads(json.dumps(value, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Diagnostic:
    code: str
    category: str
    reason: str
    operation: int | None = None

    def __post_init__(self):
        if self.category not in ("resource", "policy", "uncertainty", "unsupported"):
            raise ValueError("unknown diagnostic category")
        if not self.code or not self.reason:
            raise ValueError("diagnostics require a code and reason")


@dataclass(frozen=True)
class KernelFacts:
    backend: str
    target: dict
    operations: list[dict] = field(default_factory=list)
    regions: list[dict] = field(default_factory=list)
    ownership: list[dict] = field(default_factory=list)
    storage: list[dict] = field(default_factory=list)
    dependencies: list[dict] = field(default_factory=list)
    launch: dict = field(default_factory=dict)
    unresolved: list[Diagnostic] = field(default_factory=list)
    payload: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    version: int = FACT_VERSION

    def __post_init__(self):
        if self.version != FACT_VERSION or not self.backend or not self.target:
            raise ValueError("unsupported facts version or missing backend/target")
        # Validate without constructing and encoding a second full report in
        # the scoring path. Serialization is explicit at artifact boundaries.
        _check_json({k: v for k, v in vars(self).items() if k != "unresolved"})
        for diagnostic in self.unresolved:
            if not isinstance(diagnostic, Diagnostic):
                raise ValueError("unresolved facts require structured diagnostics")

    def to_dict(self):
        return json_value(asdict(self))

    @classmethod
    def from_dict(cls, value):
        value = json_value(value)
        value["unresolved"] = [Diagnostic(**item) for item in value.get("unresolved", [])]
        return cls(**value)


@dataclass(frozen=True)
class AnalysisReport:
    backend: str
    score: float | None
    units: str
    resources: dict
    diagnostics: list[Diagnostic]
    analysis_seconds: float
    details: dict = field(default_factory=dict)
    version: int = REPORT_VERSION

    def __post_init__(self):
        if self.version != REPORT_VERSION:
            raise ValueError("unsupported analysis report version")
        if self.score is not None and (not isfinite(self.score) or self.score < 0):
            raise ValueError("score must be finite and nonnegative or unknown")
        if not isfinite(self.analysis_seconds) or self.analysis_seconds < 0:
            raise ValueError("analysis cost must be finite and nonnegative")

    def to_dict(self):
        return json_value(asdict(self))

    @classmethod
    def from_dict(cls, value):
        value = json_value(value)
        value["diagnostics"] = [Diagnostic(**item) for item in value["diagnostics"]]
        return cls(**value)
