"""Configuration for exhaustive or top-k tile-based autotuning."""

from dataclasses import asdict, dataclass, replace

ANALYSIS_VERSION = 37


@dataclass(frozen=True)
class TileTuneConfig:
    enabled: bool = False
    mode: str = "reject"
    register_cap: int | None = None  # Optional tighter cap; known targets supply the hardware ceiling.
    max_spill_bytes: int | None = 0  # None records spills without imposing a limit.
    max_local_bytes: int | None = 0  # None records local memory without imposing a limit.
    # Override compiler-resource policy without changing analysis or selection.
    post_compile_policy: dict | None = None
    # Soft tile-demand allowance in 32-bit registers per attention computing thread.
    # Physical occupancy and post-compile limits remain strict and independent.
    attention_spill_budget_registers_per_thread: int = 0
    report_path: str | None = None
    ranking: bool = True
    top_k: int | None = None  # Select this many candidates, expanding scored or permitted-unscored boundary ties.
    strict_top_k: bool = False  # Exclude, rather than split or expand, a tie group that crosses top_k.
    alpha: float | None = None  # Strict original-pool fraction; mutually exclusive with top_k.
    exploration_fraction: float = 0.0  # Opt-in unknown-cost attempts; pure ranking remains the default.
    exploration_seed: int = 123
    device_limits: dict | None = None
    specialization: str = "auto"
    # The profile-free, kernel-family-independent memory ordering is the
    # default. Timing and occupancy models remain explicit opt-ins.
    ranking_metric: str = "memory"
    performance_model: dict | None = None
    trace_path: str | None = None  # Append intermediate analysis snapshots for manual review.
    facts_path: str | None = None  # Optional portable compiler-fact artifact.
    # Memory ranking needs only the access ledger. Enable this to additionally
    # report dependency, tile-propagation, register-liveness and shared-lifetime facts.
    memory_diagnostics: bool = False
    # Read-only one-dimensional integer parameters, keyed by PrimFunc argument
    # index. Callers must verify these values against the supplied inputs.
    input_values: dict | None = None

    def __post_init__(self):
        if self.post_compile_policy is not None:
            allowed = {"mode", "register_cap", "max_spill_bytes", "max_local_bytes"}
            if not isinstance(self.post_compile_policy, dict) or self.post_compile_policy.keys() - allowed:
                raise ValueError("post_compile_policy accepts mode, register_cap, max_spill_bytes and max_local_bytes")
            # Reuse validation for the effective compiler-only settings.
            self.compiler_resource_config()
        if self.input_values is not None and (
            not isinstance(self.input_values, dict)
            or any(
                not str(key).isdigit()
                or not isinstance(values, (list, tuple))
                or not values
                or any(type(value) is not int for value in values)
                for key, values in self.input_values.items()
            )
        ):
            raise ValueError("input_values maps parameter indices to nonempty integer vectors")
        if self.facts_path is not None and (not isinstance(self.facts_path, str) or not self.facts_path.strip()):
            raise ValueError("facts_path must be a nonempty string or None")
        if (
            isinstance(self.exploration_fraction, bool)
            or not isinstance(self.exploration_fraction, (int, float))
            or not 0 <= self.exploration_fraction <= 1
        ):
            raise ValueError("exploration_fraction must be in [0, 1]")
        if type(self.exploration_seed) is not int or self.exploration_seed < 0:
            raise ValueError("exploration_seed must be a nonnegative integer")
        if self.exploration_fraction and self.top_k is None:
            raise ValueError("exploration requires top_k")
        if self.trace_path is not None and (not isinstance(self.trace_path, str) or not self.trace_path.strip()):
            raise ValueError("trace_path must be a nonempty string or None")
        if self.specialization not in ("auto", "generic", "gemm", "attention"):
            raise ValueError("specialization must be auto, generic, gemm, or attention")
        if self.ranking_metric not in ("memory", "bound_aware", "traffic_waves", "pipeline_time"):
            raise ValueError("ranking_metric must be memory, bound_aware, traffic_waves, or pipeline_time")
        if not isinstance(self.memory_diagnostics, bool):
            raise ValueError("memory_diagnostics must be a bool")
        if self.performance_model is not None:
            from .profiling.profile_schema import validate_performance_model

            validate_performance_model(self.performance_model)
        if not isinstance(self.ranking, bool):
            raise ValueError("ranking must be a bool")
        if self.alpha is not None:
            import math

            if (
                isinstance(self.alpha, bool)
                or not isinstance(self.alpha, (int, float))
                or not math.isfinite(self.alpha)
                or not 0 < self.alpha <= 1
            ):
                raise ValueError("alpha must be finite and in (0, 1]")
            if self.top_k is not None:
                raise ValueError("alpha and top_k are mutually exclusive")
            if not self.ranking:
                raise ValueError("alpha requires ranking=True")
            if self.exploration_fraction:
                raise ValueError("alpha does not support exploration")
        if self.top_k is not None:
            if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k <= 0:
                raise ValueError("top_k must be a positive integer or None")
            if not self.ranking:
                raise ValueError("top_k requires ranking=True")
        if not isinstance(self.strict_top_k, bool):
            raise ValueError("strict_top_k must be a bool")
        if self.strict_top_k and self.top_k is None and self.alpha is None:
            raise ValueError("strict_top_k requires top_k or alpha")
        if self.strict_top_k and self.exploration_fraction:
            raise ValueError("strict_top_k does not support exploration")
        if self.device_limits is not None:
            from .src.device import DEVICE_LIMIT_FIELDS

            if not isinstance(self.device_limits, dict) or set(self.device_limits) - DEVICE_LIMIT_FIELDS:
                raise ValueError("device_limits must contain supported device resource limits")
            if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in self.device_limits.values()):
                raise ValueError("device_limits values must be positive integers")
        if self.mode not in ("reject", "report_only"):
            raise ValueError("TileTune mode must be 'reject' or 'report_only'")
        spill_budget = self.attention_spill_budget_registers_per_thread
        if isinstance(spill_budget, bool) or not isinstance(spill_budget, int) or spill_budget < 0:
            raise ValueError("attention_spill_budget_registers_per_thread must be a nonnegative integer")
        if self.ranking_metric in ("memory", "bound_aware") and (
            self.specialization not in ("auto", "generic") or spill_budget
        ):
            raise ValueError(
                "memory ranking is kernel-family independent; family specialization and attention spill allowances are unsupported"
            )
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
            raise TypeError("tiletune must be a bool, dict, or TileTuneConfig")
        return replace(config, **kwargs)

    def compiler_resource_config(self):
        """Resolve compiler-only overrides; pre-lowering uses the original config."""
        if self.post_compile_policy is None:
            return self
        return replace(self, post_compile_policy=None, **self.post_compile_policy)

    def to_cache_key_dict(self):
        values = asdict(self)
        values.pop("report_path")
        values.pop("trace_path")
        values.pop("facts_path")
        return {"analysis_version": ANALYSIS_VERSION, **values}


class TileTuneReject(Exception):
    """A pressure decision prevented lowering or benchmarking."""
