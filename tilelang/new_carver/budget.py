"""Architecture limits for register-pressure decisions, without device queries."""

from collections.abc import Mapping
import json
import re

from tvm.target import Target


# Maximum 32-bit registers per thread, not the register file capacity per SM.
# Maxwell through Blackwell: CUDA Programming Guide, technical specifications.
# https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
# https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#features-and-technical-specifications
_CUDA_255_REGISTER_ARCHS = {50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90, 100, 101, 103, 110, 120, 121}


def resolve_register_budget(config, target=None):
    """Combine a known target's hardware ceiling with an optional user cap.

    Missing and unrecognized architectures remain unknown. In particular, do
    not infer a cross-compilation target's limits from the local CUDA device.
    This ceiling does not model occupancy or warp-specific register allocation.
    """
    if target is None:
        target = Target.current(allow_none=True)
    # Constructing Target("cuda") queries device 0 when no arch is supplied.
    # Read target descriptions directly so an absent arch stays unknown.
    if isinstance(target, str):
        target = json.loads(target) if target.lstrip().startswith("{") else {"kind": target.strip()}

    if isinstance(target, Mapping):
        kind, attrs = target.get("kind"), target
    elif isinstance(target, Target):
        kind, attrs = target.kind.name, target.attrs
    else:
        kind, attrs = None, {}

    arch, hardware_cap = None, None
    if kind == "cuda":
        arch = attrs.get("arch")
        arch = str(arch) if arch is not None else None
        match = re.fullmatch(r"sm_(\d+)[af]?", arch or "")
        if match and int(match.group(1)) in _CUDA_255_REGISTER_ARCHS:
            hardware_cap = 255

    budget = config.register_cap
    source = "user register cap" if budget is not None else None
    if hardware_cap is not None and (budget is None or hardware_cap < budget):
        budget, source = hardware_cap, "architecture register limit"
    return {
        "budget": budget,
        "budget_source": source,
        "hardware_register_cap": hardware_cap,
        "target_arch": arch,
    }
