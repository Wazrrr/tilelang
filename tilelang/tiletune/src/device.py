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

    # PyTorch releases expose different subsets of cudaDeviceProp. Query absent
    # fields from the matching runtime device instead of assuming a capacity.
    # Attribute IDs are cudaDeviceAttr values from CUDA's driver_types.h.
    names = {
        "sm_count": ("multi_processor_count", 16),
        "shared_memory_per_sm": ("shared_memory_per_multiprocessor", 81),
        "shared_memory_per_block": ("shared_memory_per_block_optin", 97),
        "registers_per_sm": ("regs_per_multiprocessor", 82),
        "max_threads_per_sm": ("max_threads_per_multi_processor", 39),
        "max_threads_per_block": ("max_threads_per_block", 1),
        "warp_size": ("warp_size", 10),
        "max_blocks_per_sm": ("max_blocks_per_multi_processor", 106),
    }
    values = {}
    for key, (name, attribute) in names.items():
        value = getattr(props, name, None)
        values[key] = value if value is not None and value > 0 else get_device_attribute(attribute, device)
    return {k: int(v) for k, v in values.items() if v is not None and v > 0}
