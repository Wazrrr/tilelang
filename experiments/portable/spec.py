"""Explicit workloads, target descriptions and configuration grids.

No device detection, compiler import, or profiling occurs while planning. Preset
targets are examples: the runtime must verify the actual device before execution.
"""

from dataclasses import asdict, dataclass
from itertools import product
import re


TARGETS = {
    "ampere": {"kind": "cuda", "arch": "sm_80"},
    "hopper": {"kind": "cuda", "arch": "sm_90a"},
    "blackwell": {"kind": "cuda", "arch": "sm_100a"},
    "mi308": {"kind": "hip", "mcpu": "gfx942", "thread_warp_size": 64},
    "ascend910": {"kind": "ascendc", "arch": "Ascend910"},
}

_PARAMETERS = {
    "gemm": ({"m", "n", "k"}, {"batch", "transpose_a", "transpose_b", "epilogue"}),
    "attention": ({"batch", "heads", "sequence", "dim"}, {"causal"}),
    "kda_recurrent": ({"batch", "heads", "sequence", "dim", "value_dim"}, set()),
    "kda_chunk_o": ({"batch", "heads", "sequence", "dim", "value_dim", "chunk_size"}, set()),
    "softmax": ({"rows", "columns"}, set()),
    "rmsnorm": ({"rows", "columns"}, {"epsilon"}),
    "reduce_sum": ({"rows", "columns"}, set()),
    "elementwise": ({"rows", "columns"}, set()),
}

_DTYPES = ("float16", "bfloat16", "float32", "float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")


@dataclass(frozen=True)
class Workload:
    name: str
    op: str
    parameters: dict
    dtype: str = "float16"
    configs: list[dict] | None = None

    def __post_init__(self):
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
            elif key == "epilogue":
                if value not in ("none", "bias", "bias_relu"):
                    raise ValueError("epilogue must be none, bias, or bias_relu")
            elif key == "epsilon":
                from math import isfinite

                if isinstance(value, bool) or not isinstance(value, float | int) or not isfinite(value) or value <= 0:
                    raise ValueError("epsilon must be finite and positive")
            elif type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.dtype not in _DTYPES:
            raise ValueError(f"Unsupported workload dtype {self.dtype}")
        if self.op in ("attention", "kda_chunk_o") and self.dtype not in ("float16", "bfloat16"):
            raise ValueError(f"{self.op} supports float16 and bfloat16")
        if self.op != "gemm" and self.dtype.startswith("float8"):
            raise ValueError("FP8 is currently an explicit GEMM workload")
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
        return asdict(self)


def _grid(**axes):
    return [dict(zip(axes, values)) for values in product(*axes.values())]


def configurations(workload, device):
    if device.configs and workload.name in device.configs:
        return [dict(c) for c in device.configs[workload.name]]
    if workload.configs is not None:
        return [dict(c) for c in workload.configs]
    # Search spaces are experimental inputs, not hardware capability claims.
    # Native GPU kernels use the same grid across architectures for comparison.
    if workload.op == "gemm":
        return _grid(block_m=[32, 64, 128], block_n=[32, 64, 128], block_k=[32, 64], stages=[0, 2, 3], threads=[128, 256])
    if workload.op == "attention":
        return _grid(block_M=[64, 128, 32], block_N=[64, 128, 32], num_stages=[0, 2, 3], threads=[128, 256])
    if workload.op == "kda_recurrent":
        return _grid(block_v=[16, 32, 64], threads=[128, 256])
    if workload.op == "kda_chunk_o":
        return _grid(block_k=[32, 64], block_v=[32, 64], stages=[0, 2, 3], threads=[128, 256])
    return _grid(block_rows=[1, 2, 4], threads=[128, 256])


def default_workloads(smoke=False):
    m = 128 if smoke else 2048
    sequence = 128 if smoke else 2048
    rows = 32 if smoke else 4096
    cols = 128 if smoke else 4096
    gemm = dict(m=m, n=m, k=m)
    attention = dict(batch=1, heads=2 if smoke else 16, sequence=sequence, dim=64)
    kda = dict(batch=1, heads=2 if smoke else 8, sequence=16 if smoke else 256, dim=32 if smoke else 64, value_dim=32 if smoke else 64)
    return [
        Workload("gemm_nn", "gemm", gemm),
        Workload("gemm_nt", "gemm", dict(gemm, transpose_b=True)),
        Workload("gemm_tn", "gemm", dict(gemm, transpose_a=True)),
        Workload("gemm_batched", "gemm", dict(gemm, batch=4)),
        Workload("gemm_bias_relu", "gemm", dict(gemm, epilogue="bias_relu")),
        Workload("gemm_bf16", "gemm", gemm, "bfloat16"),
        Workload("gemm_fp8", "gemm", gemm, "float8_e4m3fn"),
        Workload("gemm_fp8_fnuz", "gemm", gemm, "float8_e4m3fnuz"),
        Workload("gemm_tall", "gemm", dict(m=m * 4, n=max(32, m // 4), k=m)),
        Workload("gemm_wide", "gemm", dict(m=max(32, m // 4), n=m * 4, k=m)),
        Workload("flashattention", "attention", attention),
        Workload("flashattention_causal", "attention", dict(attention, causal=True)),
        Workload("flashattention_bf16", "attention", attention, "bfloat16"),
        Workload("kda_recurrent", "kda_recurrent", kda),
        Workload("kda_chunk_o", "kda_chunk_o", dict(kda, sequence=64 if smoke else 1024, chunk_size=32 if smoke else 64)),
        *[Workload(op, op, dict(rows=rows, columns=cols)) for op in ("softmax", "rmsnorm", "reduce_sum", "elementwise")],
    ]


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
        missing = set(device.configs or {}) - names
        if missing:
            raise ValueError(f"device {device.name} has configuration overrides for unknown workloads: {sorted(missing)}")
    return devices, workloads


def support_reason(workload, device):
    kind = device.target["kind"]
    if kind not in ("cuda", "hip"):
        return (
            None
            if device.worker
            else "requires an external TileLang-Ascend worker; this checkout has no Ascend compiler or core/storage model"
        )
    if workload.dtype.startswith("float8"):
        if kind == "hip":
            return None if workload.dtype.endswith("fnuz") else "MI308 FP8 requires an explicit FNUZ workload dtype"
        if workload.dtype.endswith("fnuz"):
            return "FNUZ workloads require an AMD backend; CUDA FP8 uses the FN/E5M2 formats"
        match = re.fullmatch(r"sm_(\d+)[af]?", device.target["arch"])
        if not match or int(match[1]) < 89:
            return "target has no supported FP8 matrix instructions"
    return None
