"""Read CUDA occupancy directly from the installed NVML driver library.

The small ctypes declarations follow CUDA's include/nvml.h. No CUDA context is
created. Keeping queries in-process avoids spawning nvidia-smi twice per poll.
"""

import ctypes as C
from datetime import datetime, timezone
from pathlib import Path
import time


class Memory(C.Structure):
    _fields_ = [(name, C.c_ulonglong) for name in ("total", "free", "used")]


class Utilization(C.Structure):
    _fields_ = [("gpu", C.c_uint), ("memory", C.c_uint)]


class Process(C.Structure):
    # nvmlDeviceGetComputeRunningProcesses_v3 uses nvmlProcessInfo_t, the v2
    # struct (pid, memory, GPU/compute instance IDs), not protected-memory info.
    _fields_ = [("pid", C.c_uint), ("usedGpuMemory", C.c_ulonglong), ("gpuInstanceId", C.c_uint), ("computeInstanceId", C.c_uint)]


_library = None


def library():
    global _library
    if _library is None:
        lib = C.CDLL("libnvidia-ml.so.1")
        lib.nvmlErrorString.restype = C.c_char_p
        check(lib, lib.nvmlInit_v2())
        _library = lib
    return _library


def check(lib, result):
    if result:
        raise RuntimeError("NVML: " + lib.nvmlErrorString(result).decode())


def snapshot():
    started = time.monotonic()
    lib = library()
    count = C.c_uint()
    check(lib, lib.nvmlDeviceGetCount_v2(C.byref(count)))
    gpus, processes = [], []
    for i in range(count.value):
        handle = C.c_void_p()
        check(lib, lib.nvmlDeviceGetHandleByIndex_v2(C.c_uint(i), C.byref(handle)))
        uuid, name = C.create_string_buffer(160), C.create_string_buffer(160)
        check(lib, lib.nvmlDeviceGetUUID(handle, uuid, len(uuid)))
        check(lib, lib.nvmlDeviceGetName(handle, name, len(name)))
        major, minor = C.c_int(), C.c_int()
        check(lib, lib.nvmlDeviceGetCudaComputeCapability(handle, C.byref(major), C.byref(minor)))
        memory, utilization = Memory(), Utilization()
        check(lib, lib.nvmlDeviceGetMemoryInfo(handle, C.byref(memory)))
        check(lib, lib.nvmlDeviceGetUtilizationRates(handle, C.byref(utilization)))
        values = {}
        for field, function, extra, divisor in (
            ("clocks.sm", lib.nvmlDeviceGetClockInfo, [1], 1),
            ("temperature.gpu", lib.nvmlDeviceGetTemperature, [0], 1),
            ("power.draw", lib.nvmlDeviceGetPowerUsage, [], 1000),
        ):
            value = C.c_uint()
            check(lib, function(handle, *extra, C.byref(value)))
            values[field] = str(value.value / divisor)
        uid = uuid.value.decode()
        gpus.append(
            dict(
                index=str(i),
                uuid=uid,
                name=name.value.decode(),
                compute_cap=f"{major.value}.{minor.value}",
                **{"utilization.gpu": str(utilization.gpu), "memory.used": str(memory.used // 1048576), **values},
            )
        )
        size = C.c_uint(32)
        while True:
            entries = (Process * size.value)()
            result = lib.nvmlDeviceGetComputeRunningProcesses_v3(handle, C.byref(size), entries)
            if result == 7:  # Process list grew beyond the allocated capacity.
                size.value += 16
                continue
            check(lib, result)
            break
        for entry in entries[: size.value]:
            try:
                process_name = Path(f"/proc/{entry.pid}/comm").read_text().strip()
            except (FileNotFoundError, PermissionError):
                process_name = "unavailable"
            processes.append(
                dict(gpu_uuid=uid, pid=str(entry.pid), process_name=process_name, used_memory=str(entry.usedGpuMemory // 1048576))
            )
    return dict(
        timestamp=datetime.now(timezone.utc).isoformat(),
        gpus=gpus,
        processes=processes,
        snapshot_backend="nvml",
        duration_seconds=time.monotonic() - started,
    )
