"""Explicit workloads, target descriptions and configuration grids.

No device detection, compiler import, or profiling occurs while planning. Preset
targets are examples: the runtime must verify the actual device before execution.
"""

from dataclasses import asdict, dataclass
import re

from .spaces import PRESETS


TARGETS = {
    "ampere": {"kind": "cuda", "arch": "sm_80"},
    "hopper": {"kind": "cuda", "arch": "sm_90a"},
    "blackwell": {"kind": "cuda", "arch": "sm_100a"},
    "mi355x": {"kind": "hip", "mcpu": "gfx950", "thread_warp_size": 64},
    "ascend910b": {"kind": "ascendc", "arch": "Ascend910B"},
    "mi308": {"kind": "hip", "mcpu": "gfx942", "thread_warp_size": 64},
    "ascend910": {"kind": "ascendc", "arch": "Ascend910"},
}

_PARAMETERS = {
    "gemm": ({"m", "n", "k"}, {"batch", "transpose_a", "transpose_b", "epilogue"}),
    "grouped_gemm": ({"batch_sizes", "n", "k"}, {"transpose_b"}),
    "attention": ({"batch", "heads", "sequence", "dim"}, {"causal"}),
    "kda_chunk_o": ({"batch", "heads", "sequence", "dim", "value_dim", "chunk_size"}, set()),
    "gemm_fp8": ({"m", "n", "k"}, {"transpose_b"}),
}

_DTYPES = ("float16", "bfloat16", "float32", "float8_e4m3fn", "float8_e5m2")


@dataclass(frozen=True)
class Workload:
    name: str
    op: str
    parameters: dict
    dtype: str = "float16"
    configs: list[dict] | None = None
    config_space: str | None = None

    def __post_init__(self):
        if self.config_space is None:
            object.__setattr__(self, "config_space", "expanded")
        if self.config_space not in PRESETS:
            raise ValueError("experiments have one configuration space: expanded")
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", self.name):
            raise ValueError("workload name must be a single safe path component")
        if self.op not in _PARAMETERS:
            raise ValueError(f"Unknown operation {self.op!r}; choose from {list(_PARAMETERS)}")
        required, optional = _PARAMETERS[self.op]
        if not required <= self.parameters.keys() or self.parameters.keys() - required - optional:
            raise ValueError(f"{self.op} requires {sorted(required)}; optional parameters: {sorted(optional)}")
        for key, value in self.parameters.items():
            if key in {"transpose_a", "transpose_b", "causal"}:
                if not isinstance(value, bool):
                    raise ValueError(f"{key} must be a bool")
            elif key == "batch_sizes":
                if not isinstance(value, list) or not value or any(type(size) is not int or size <= 0 for size in value):
                    raise ValueError("batch_sizes must be a nonempty list of positive integers")
            elif key == "epilogue":
                if value not in ("none", "bias", "bias_relu"):
                    raise ValueError("epilogue must be none, bias, or bias_relu")
            elif type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.dtype not in _DTYPES:
            raise ValueError(f"Unsupported workload dtype {self.dtype}")
        if self.op == "gemm_fp8" and self.dtype not in ("float8_e4m3fn", "float8_e5m2"):
            raise ValueError("gemm_fp8 requires float8_e4m3fn or float8_e5m2")
        if self.op != "gemm_fp8" and self.dtype.startswith("float8"):
            raise ValueError("FP8 inputs require the gemm_fp8 example")
        if self.op in ("attention", "kda_chunk_o", "grouped_gemm") and self.dtype not in ("float16", "bfloat16"):
            raise ValueError(f"{self.op} supports float16 and bfloat16")
        if self.op == "kda_chunk_o" and self.parameters["sequence"] % self.parameters["chunk_size"]:
            raise ValueError("kda_chunk_o requires complete chunks")
        if self.configs is not None and (not self.configs or any(not isinstance(c, dict) or not c for c in self.configs)):
            raise ValueError("configs must be a nonempty list of nonempty dictionaries")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Device:
    name: str
    target: dict
    device_limits: dict | None = None
    performance_model: dict | None = None
    # Reusable profile bundles, keyed by input dtype. Paths in a manifest are
    # resolved relative to that manifest by the CLI before freezing requests.
    profiles: dict[str, str] | None = None
    # Per-workload target-specific grids, e.g. Cube/L1 knobs instead of GPU
    # thread/stage knobs. Mathematical parameters remain in Workload.
    configs: dict[str, list[dict]] | None = None
    # Execute this argv in the target's own environment, with request/result
    # paths appended. Useful for separate Ascend TVM/CANN installations or SSH.
    worker: list[str] | None = None
    worker_cwd: str | None = None
    # Original pool indices frozen by a named suite, independently per case.
    subsets: dict[str, list[int]] | None = None
    expected_device_pattern: str | None = None

    def __post_init__(self):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", self.name):
            raise ValueError("device name must be a single safe path component")
        if not isinstance(self.target, dict) or not self.target.get("kind"):
            raise ValueError("device target requires an explicit kind")
        kind = self.target["kind"]
        if kind not in ("cuda", "hip", "ascendc", "pto", "npuir"):
            raise ValueError(f"Unsupported experiment backend {kind}")
        if not self.target.get("mcpu" if kind == "hip" else "arch"):
            raise ValueError("device target requires an explicit architecture")
        if self.worker is not None and (not self.worker or any(not isinstance(a, str) or not a for a in self.worker)):
            raise ValueError("worker must be a nonempty argv list")
        if self.expected_device_pattern is not None:
            re.compile(self.expected_device_pattern)
        if self.subsets is not None and any(
            not isinstance(indices, list)
            or not indices
            or len(set(indices)) != len(indices)
            or any(type(i) is not int or i < 0 for i in indices)
            for indices in self.subsets.values()
        ):
            raise ValueError("subsets require unique nonnegative original indices")
        if self.profiles is not None:
            if self.performance_model is not None:
                raise ValueError("supply profiles or performance_model, not both")
            if (
                not isinstance(self.profiles, dict)
                or not self.profiles
                or any(key not in _DTYPES for key in self.profiles)
                or any(not isinstance(v, str) or not v for v in self.profiles.values())
            ):
                raise ValueError("profiles must map input dtypes to nonempty paths")
        if self.configs is not None and (
            not isinstance(self.configs, dict)
            or any(
                not isinstance(name, str) or not isinstance(grid, list) or not grid or any(not isinstance(c, dict) or not c for c in grid)
                for name, grid in self.configs.items()
            )
        ):
            raise ValueError("device configs must map workload names to nonempty configuration grids")

    def to_dict(self):
        values = asdict(self)
        # Preserve the byte-level request identity of archived v1 manifests.
        for key in ("subsets", "expected_device_pattern"):
            if values[key] is None:
                values.pop(key)
        return values


def configurations(workload, device):
    return configuration_space(workload, device)["configs"]


def configuration_space(workload, device):
    """Return the deterministic pool and its timing-independent generation audit."""
    from .spaces import audit_space
    from experiments.families import family_module

    pool = family_module(workload.op, "spaces").get_configs()
    supplied = (device.configs or {}).get(workload.name, workload.configs)
    if supplied is not None and device.target["kind"] in ("cuda", "hip") and any(c not in pool for c in supplied):
        raise ValueError(f"{workload.op.upper()} explicit configurations must be a subset of the expanded example pool")
    if supplied is not None:
        return audit_space(workload, supplied, explicit=True, retained_current_count=len(supplied))
    return audit_space(workload, pool)


def default_workloads(smoke=False):
    """Use the family-owned final cases, or development shapes for smoke checks."""
    from experiments.families import DEFAULT_OPS, family_module

    return [w for op in DEFAULT_OPS for w in family_module(op, "cases").cases(holdout=not smoke)]


def load_manifest(data):
    if not isinstance(data, dict) or data.get("version") != 1 or set(data) != {"version", "devices", "workloads"}:
        raise ValueError("manifest requires version=1, devices, and workloads")
    devices = [Device(**d) for d in data["devices"]]
    workloads = [Workload(**w) for w in data["workloads"]]
    for values in (devices, workloads):
        if not values or len({v.name for v in values}) != len(values):
            raise ValueError("manifest lists must be nonempty with unique names")
    names = {workload.name for workload in workloads}
    for device in devices:
        missing = (set(device.configs or {}) | set(device.subsets or {})) - names
        if missing:
            raise ValueError(f"device {device.name} has configuration overrides for unknown workloads: {sorted(missing)}")
    return devices, workloads


def support_reason(workload, device):
    if workload.op == "gemm_fp8":
        from experiments.gemm_fp8.spaces import support_reason as fp8_support_reason

        reason = fp8_support_reason(workload, device)
        if reason:
            return reason
    if workload.op in ("gemm", "grouped_gemm"):
        from experiments.families import family_module

        reason = family_module(workload.op, "spaces").support_reason(workload)
        if reason:
            return reason
    kind = device.target["kind"]
    if kind not in ("cuda", "hip"):
        return (
            None
            if device.worker
            else "requires an external TileLang-Ascend worker; this checkout has no Ascend compiler or core/storage model"
        )
    return None
