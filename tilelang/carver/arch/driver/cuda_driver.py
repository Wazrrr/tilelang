from __future__ import annotations
import ctypes
import sys

try:
    import torch.cuda._CudaDeviceProperties as _CudaDeviceProperties
except ImportError:
    _CudaDeviceProperties = type("DummyCudaDeviceProperties", (), {})


class cudaDeviceAttrNames:
    r"""
    refer to https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g49e2f8c2c0bd6fe264f2fc970912e5cd
    """

    cudaDevAttrMaxThreadsPerBlock: int = 1
    cudaDevAttrMaxBlockDimX: int = 2
    cudaDevAttrMaxBlockDimY: int = 3
    cudaDevAttrMaxBlockDimZ: int = 4
    cudaDevAttrMaxGridDimX: int = 5
    cudaDevAttrMaxGridDimY: int = 6
    cudaDevAttrMaxGridDimZ: int = 7
    cudaDevAttrMaxSharedMemoryPerBlock: int = 8
    cudaDevAttrMaxRegistersPerBlock: int = 12
    cudaDevAttrMultiProcessorCount: int = 16
    cudaDevAttrMaxThreadsPerMultiProcessor: int = 39
    cudaDevAttrMaxSharedMemoryPerMultiprocessor: int = 81
    cudaDevAttrMaxRegistersPerMultiprocessor: int = 82
    cudaDevAttrCooperativeLaunch: int = 95
    cudaDevAttrMaxSharedMemoryPerBlockOptin: int = 97
    cudaDevAttrMaxBlocksPerMultiprocessor: int = 106
    cudaDevAttrMaxPersistingL2CacheSize: int = 108
    cudaDevAttrReservedSharedMemoryPerBlock: int = 111


def get_cuda_device_properties(device_id: int = 0) -> _CudaDeviceProperties | None:
    try:
        import torch.cuda

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(torch.device(device_id))
    except ImportError:
        return None


def get_device_name(device_id: int = 0) -> str | None:
    prop = get_cuda_device_properties(device_id)
    if prop:
        return prop.name


def get_shared_memory_per_block(device_id: int = 0, format: str = "bytes") -> int | None:
    assert format in ["bytes", "kb", "mb"], "Invalid format. Must be one of: bytes, kb, mb"
    prop = get_cuda_device_properties(device_id)
    if prop is None:
        raise RuntimeError("Failed to get device properties.")
    shared_mem = int(prop.shared_memory_per_block)
    if format == "bytes":
        return shared_mem
    elif format == "kb":
        return shared_mem // 1024
    elif format == "mb":
        return shared_mem // (1024 * 1024)
    else:
        raise RuntimeError("Invalid format. Must be one of: bytes, kb, mb")


def _load_cudart():
    """Load the CUDA runtime library, searching for the version-suffixed DLL on Windows."""
    if sys.platform == "win32":
        for ver in ("13", "12", "110", "11"):
            try:
                return ctypes.windll.LoadLibrary(f"cudart64_{ver}.dll")
            except OSError:
                continue
        raise OSError("Cannot find cudart64_*.dll")
    return ctypes.cdll.LoadLibrary("libcudart.so")


def get_device_attribute(attr: int, device_id: int = 0) -> int:
    try:
        libcudart = _load_cudart()

        value = ctypes.c_int()
        cudaDeviceGetAttribute = libcudart.cudaDeviceGetAttribute
        cudaDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        cudaDeviceGetAttribute.restype = ctypes.c_int

        ret = cudaDeviceGetAttribute(ctypes.byref(value), attr, device_id)
        if ret != 0:
            raise RuntimeError(f"cudaDeviceGetAttribute failed with error {ret}")

        return value.value
    except Exception as e:
        print(f"Error getting device attribute: {e}")
        return None


def get_max_dynamic_shared_size_bytes(device_id: int = 0, format: str = "bytes") -> int | None:
    """
    Get the maximum opt-in dynamic shared memory size per block.
    """
    assert format in ["bytes", "kb", "mb"], "Invalid format. Must be one of: bytes, kb, mb"
    shared_mem = get_device_attribute(cudaDeviceAttrNames.cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id)
    if format == "bytes":
        return shared_mem
    elif format == "kb":
        return shared_mem // 1024
    elif format == "mb":
        return shared_mem // (1024 * 1024)
    else:
        raise RuntimeError("Invalid format. Must be one of: bytes, kb, mb")


def get_persisting_l2_cache_max_size(device_id: int = 0) -> int:
    prop = get_device_attribute(cudaDeviceAttrNames.cudaDevAttrMaxPersistingL2CacheSize, device_id)
    return prop


def get_num_sms(device_id: int = 0) -> int:
    """
    Get the number of streaming multiprocessors (SMs) on the CUDA device.

    Args:
        device_id (int, optional): The CUDA device ID. Defaults to 0.

    Returns:
        int: The number of SMs on the device.

    Raises:
        RuntimeError: If unable to get the device properties.
    """
    prop = get_cuda_device_properties(device_id)
    if prop is None:
        raise RuntimeError("Failed to get device properties.")
    return prop.multi_processor_count


def get_registers_per_block(device_id: int = 0) -> int:
    """
    Get the maximum number of 32-bit registers available per block.
    """
    prop = get_device_attribute(
        cudaDeviceAttrNames.cudaDevAttrMaxRegistersPerBlock,
        device_id,
    )
    return prop


def get_max_threads_per_block(device_id: int = 0) -> int | None:
    return get_device_attribute(cudaDeviceAttrNames.cudaDevAttrMaxThreadsPerBlock, device_id)


def get_max_block_dims(device_id: int = 0) -> tuple[int | None, int | None, int | None]:
    return (
        get_device_attribute(cudaDeviceAttrNames.cudaDevAttrMaxBlockDimX, device_id),
        get_device_attribute(cudaDeviceAttrNames.cudaDevAttrMaxBlockDimY, device_id),
        get_device_attribute(cudaDeviceAttrNames.cudaDevAttrMaxBlockDimZ, device_id),
    )


def get_max_grid_dims(device_id: int = 0) -> tuple[int | None, int | None, int | None]:
    return (
        get_device_attribute(cudaDeviceAttrNames.cudaDevAttrMaxGridDimX, device_id),
        get_device_attribute(cudaDeviceAttrNames.cudaDevAttrMaxGridDimY, device_id),
        get_device_attribute(cudaDeviceAttrNames.cudaDevAttrMaxGridDimZ, device_id),
    )


def get_shared_memory_per_multiprocessor(device_id: int = 0) -> int | None:
    return get_device_attribute(cudaDeviceAttrNames.cudaDevAttrMaxSharedMemoryPerMultiprocessor, device_id)


def get_registers_per_multiprocessor(device_id: int = 0) -> int | None:
    return get_device_attribute(cudaDeviceAttrNames.cudaDevAttrMaxRegistersPerMultiprocessor, device_id)


def get_max_threads_per_multiprocessor(device_id: int = 0) -> int | None:
    return get_device_attribute(cudaDeviceAttrNames.cudaDevAttrMaxThreadsPerMultiProcessor, device_id)


def get_max_blocks_per_multiprocessor(device_id: int = 0) -> int | None:
    return get_device_attribute(cudaDeviceAttrNames.cudaDevAttrMaxBlocksPerMultiprocessor, device_id)


def get_reserved_shared_memory_per_block(device_id: int = 0) -> int | None:
    return get_device_attribute(cudaDeviceAttrNames.cudaDevAttrReservedSharedMemoryPerBlock, device_id)


def get_cooperative_launch_support(device_id: int = 0) -> bool | None:
    value = get_device_attribute(cudaDeviceAttrNames.cudaDevAttrCooperativeLaunch, device_id)
    return None if value is None else bool(value)
