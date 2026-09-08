"""Configuration for exhaustive, pressure-only autotuning."""

from dataclasses import asdict, dataclass, replace

ANALYSIS_VERSION = 15


@dataclass(frozen=True)
class CarverConfig:
    enabled: bool = False
    mode: str = "reject"
    register_cap: int | None = None  # Optional tighter cap; known targets supply the hardware ceiling.
    max_spill_bytes: int | None = 0  # None records spills without imposing a limit.
    max_local_bytes: int | None = 0  # None records local memory without imposing a limit.
    # Soft tile-demand allowance in 32-bit registers per attention computing thread.
    # Physical occupancy and post-compile limits remain strict and independent.
    attention_spill_budget_registers_per_thread: int = 0
    report_path: str | None = None
    ranking: bool = True
    device_limits: dict | None = None
    specialization: str = "auto"
    ranking_metric: str = "traffic_waves"
    performance_model: dict | None = None
    trace_path: str | None = None  # Append intermediate analysis snapshots for manual review.

    def __post_init__(self):
        if self.trace_path is not None and (not isinstance(self.trace_path, str) or not self.trace_path.strip()):
            raise ValueError("trace_path must be a nonempty string or None")
        if self.specialization not in ("auto", "generic", "gemm", "attention"):
            raise ValueError("specialization must be auto, generic, gemm, or attention")
        if self.ranking_metric not in ("traffic_waves", "pipeline_time"):
            raise ValueError("ranking_metric must be traffic_waves or pipeline_time")
        if self.performance_model is not None:
            from .profile_schema import validate_performance_model

            validate_performance_model(self.performance_model)
        if not isinstance(self.ranking, bool):
            raise ValueError("ranking must be a bool")
        if self.device_limits is not None:
            from .cost import DEVICE_LIMIT_FIELDS

            if not isinstance(self.device_limits, dict) or set(self.device_limits) - DEVICE_LIMIT_FIELDS:
                raise ValueError("device_limits must contain supported device resource limits")
            if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in self.device_limits.values()):
                raise ValueError("device_limits values must be positive integers")
        if self.mode not in ("reject", "report_only"):
            raise ValueError("Carver mode must be 'reject' or 'report_only'")
        spill_budget = self.attention_spill_budget_registers_per_thread
        if isinstance(spill_budget, bool) or not isinstance(spill_budget, int) or spill_budget < 0:
            raise ValueError("attention_spill_budget_registers_per_thread must be a nonnegative integer")
        for name in ("register_cap", "max_spill_bytes", "max_local_bytes"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < (1 if name == "register_cap" else 0):
                raise ValueError(f"{name} must be a {'positive' if name == 'register_cap' else 'nonnegative'} integer or None")

    @classmethod
    def from_value(cls, value=None, **kwargs):
        if isinstance(value, cls):
            config = value
        elif isinstance(value, dict):
            config = cls(**{"enabled": True, **value})
        elif value is None or isinstance(value, bool):
            config = cls(enabled=bool(kwargs) if value is None else value)
        else:
            raise TypeError("carver must be a bool, dict, or CarverConfig")
        return replace(config, **kwargs)

    def to_cache_key_dict(self):
        values = asdict(self)
        values.pop("report_path")
        values.pop("trace_path")
        return {"analysis_version": ANALYSIS_VERSION, **values}


class CarverReject(Exception):
    """A pressure decision prevented lowering or benchmarking."""
