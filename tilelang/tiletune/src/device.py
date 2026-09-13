"""Target register capabilities and explicit current-device capacity queries."""

from ..targets import resolve_target

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


def target_register_limits(target=None):
    """Resolve target architecture and hardware registers without a device query."""
    model = resolve_target(target)
    return model.arch, model.register_cap


def query_device_limits(target=None):
    """Query only a matching device; absent capacities remain absent.

    HIP property availability varies by PyTorch/ROCm version. Callers can supply
    the missing measured capacities through device_limits. CUDA driver attribute
    numbers are never passed to HIP.
    """
    import torch

    if not torch.cuda.is_available():
        return None
    model = resolve_target(target)
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    if torch.version.hip:
        from tilelang.rocm.target import normalize_rocm_arch

        if model.kind != "hip" or model.arch is None or model.arch != normalize_rocm_arch(getattr(props, "gcnArchName", None)):
            return None
        names = {
            "sm_count": "multi_processor_count",
            "shared_memory_per_sm": "shared_memory_per_multiprocessor",
            "shared_memory_per_block": "shared_memory_per_block",
            "registers_per_sm": "regs_per_multiprocessor",
            "max_threads_per_sm": "max_threads_per_multi_processor",
            "max_threads_per_block": "max_threads_per_block",
            "max_blocks_per_sm": "max_blocks_per_multi_processor",
            "warp_size": "warp_size",
        }
        values = {key: getattr(props, name, None) for key, name in names.items()}
        return {key: int(value) for key, value in values.items() if value is not None and value > 0}
    if model.kind != "cuda" or model.arch is None or model.arch.removeprefix("sm_").rstrip("af") != f"{props.major}{props.minor}":
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
