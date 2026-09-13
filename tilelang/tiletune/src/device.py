"""Target register capabilities and explicit current-device capacity queries."""

from collections.abc import Mapping
import json
import re
from tvm.target import Target

DEVICE_LIMIT_FIELDS = {
    "sm_count",
    "shared_memory_per_sm",
    "shared_memory_per_block",
    "registers_per_sm",
    "max_threads_per_sm",
    "max_blocks_per_sm",
    "max_threads_per_block",
    "warp_size",
}

# Maximum 32-bit registers per thread, not the register file capacity per SM.
# CUDA Programming Guide: technical specifications, Maxwell through Blackwell.
# https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
_CUDA_255_REGISTER_ARCHS = {50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90, 100, 101, 103, 110, 120, 121}


def target_register_limits(target=None):
    """Resolve target architecture and hardware registers without a device query."""
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

    return arch, hardware_cap


def query_device_limits(target=None):
    """Read the current CUDA device only when it matches the compilation target."""
    import torch

    if not torch.cuda.is_available():
        return None
    arch = target_register_limits(target)[0]
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    if arch is None or arch.removeprefix("sm_").rstrip("af") != f"{props.major}{props.minor}":
        return None
    from tilelang.carver.arch.driver.cuda_driver import get_device_attribute

    values = {
        "sm_count": props.multi_processor_count,
        "shared_memory_per_sm": props.shared_memory_per_multiprocessor,
        "shared_memory_per_block": props.shared_memory_per_block_optin,
        "registers_per_sm": props.regs_per_multiprocessor,
        "max_threads_per_sm": props.max_threads_per_multi_processor,
        "max_threads_per_block": props.max_threads_per_block,
        "warp_size": props.warp_size,
        # cudaDevAttrMaxBlocksPerMultiprocessor; keep TileTune's extra query out
        # of the unmodified legacy Carver driver and policy.
        "max_blocks_per_sm": get_device_attribute(106, device),
    }
    return {k: int(v) for k, v in values.items() if v is not None and v > 0}
