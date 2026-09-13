"""Offline target identity and the boundaries of the implemented hardware models.

Architecture names select semantics, never device capacities or measured rates.
CUDA SMs and HIP CUs use the existing block/wave report vocabulary. Ascend's
Cube/Vector cores and L0/L1/UB storage require a separate execution model.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
import re


@dataclass(frozen=True)
class TargetModel:
    kind: str | None
    arch: str | None
    architecture: str
    subgroup_size: int | None
    register_cap: int | None
    block_execution: bool
    native_tuner: bool

    def to_dict(self):
        return asdict(self)

    def compiler_target(self):
        if self.kind == "cuda" and self.arch:
            return {"kind": "cuda", "arch": self.arch}
        if self.kind == "hip" and self.arch:
            return {"kind": "hip", "mcpu": self.arch, "thread_warp_size": self.subgroup_size}
        raise ValueError(f"No native TileTune compiler target for {self.kind}:{self.arch}")


# Maximum 32-bit registers per CUDA thread. This is not an SM capacity.
_CUDA_REGISTER_ARCHS = {50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90, 100, 101, 103, 110, 120, 121}


def resolve_target(target=None):
    """Describe a target without constructing a target or querying a device."""
    from tvm.target import Target

    if target is None:
        target = Target.current(allow_none=True)
    if isinstance(target, str):
        if target.lstrip().startswith("{"):
            target = json.loads(target)
        else:
            parts = target.split()
            attrs = {"kind": parts[0]} if parts else {}
            for part in parts[1:]:
                key, separator, value = part.lstrip("-").partition("=")
                if not separator:
                    raise ValueError("Target options must use -name=value or a target dictionary")
                attrs[key] = value
            target = attrs
    if isinstance(target, Mapping):
        kind, attrs = target.get("kind"), target
    elif isinstance(target, Target):
        kind, attrs = target.kind.name, target.attrs
    elif target is None:
        kind, attrs = None, {}
    else:
        raise TypeError("target must be a target description, TVM Target, or None")
    arch = attrs.get("mcpu") if kind == "hip" else attrs.get("arch")
    arch = str(arch) if arch is not None else None
    if kind == "cuda":
        match = re.fullmatch(r"sm_(\d+)[af]?", arch or "")
        version = int(match[1]) if match else None
        family = (
            "ampere"
            if version in (80, 86, 87)
            else "hopper"
            if version == 90
            else "blackwell"
            if version in (100, 101, 103, 110, 120, 121)
            else "cuda"
        )
        return TargetModel(kind, arch, family, 32, 255 if version in _CUDA_REGISTER_ARCHS else None, True, True)
    if kind == "hip":
        from tilelang.rocm.target import normalize_rocm_arch, rocm_warp_size_for_arch

        arch = normalize_rocm_arch(arch)
        return TargetModel(kind, arch, "cdna3" if arch == "gfx942" else "rocm", rocm_warp_size_for_arch(arch), None, True, True)
    if kind in ("ascend", "ascendc", "pto", "npuir"):
        return TargetModel(kind, arch, "ascend", None, None, False, False)
    return TargetModel(kind, arch, "unknown", None, None, False, False)


def current_target():
    """Explicit runtime detection; PyTorch exposes ROCm through torch.cuda."""
    import torch

    if not torch.cuda.is_available():
        raise ValueError("No CUDA or ROCm device is available; Ascend runs use a separate worker environment")
    if torch.version.hip:
        from tilelang.rocm.target import normalize_rocm_arch

        arch = normalize_rocm_arch(getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), "gcnArchName", None))
        if arch is None:
            raise ValueError("ROCm did not report the current device's gcnArchName")
        return resolve_target({"kind": "hip", "mcpu": arch}).compiler_target()
    major, minor = torch.cuda.get_device_capability()
    suffix = "a" if major in (9, 10, 11) else ""
    return {"kind": "cuda", "arch": f"sm_{major}{minor}{suffix}"}
