"""Allocation sizes and layout-based register estimates.

This stage describes storage without proving that its elements are simultaneously
live. Buffer objects key the internal model; names are only report labels.
"""

from dataclasses import dataclass
from math import ceil, prod

from tilelang import tvm


@dataclass
class RegisterStorage:
    """Report entries plus the subset with per-thread allocation estimates."""

    logical_storage: list[dict]
    modeled_buffers: dict


def analyze_register_storage(col):
    """Describe each allocation using its scope, dtype and optional layout."""
    from .analysis import _int

    storage = []
    modeled = {}
    for buffer in col.buffers:
        shape = [_int(x) for x in buffer.shape]
        size = prod(shape) if all(x is not None for x in shape) else None
        dtype = tvm.DataType(buffer.dtype)
        entry = {
            "buffer": buffer.name,
            "scope": buffer.scope(),
            "dtype": str(buffer.dtype),
            "logical_elements": size,
            "logical_bits": size * dtype.bits * dtype.lanes if size is not None else None,
            "modeled_registers_per_thread": None,
            "evidence": [],
        }
        layout = col.layouts.get(buffer.data)
        if buffer.scope() == "local.fragment" and layout is not None and hasattr(layout, "get_thread_size") and size is not None:
            threads = _int(layout.get_thread_size())
            # An explicit layout fixes ownership. The allocation's index extent
            # remains an estimate: holes and unused elements may be eliminated.
            local_shape = [_int(x) for x in layout.get_output_shape()]
            if threads and all(x is not None for x in local_shape):
                slots = prod(local_shape)
                entry["modeled_registers_per_thread"] = {
                    "lower": ceil(slots * dtype.bits * dtype.lanes / 32),
                    "upper": slots * ceil(dtype.bits * dtype.lanes / 32),
                }
                entry["replication"] = _int(layout.replicate_size)
                entry["computing_threads"] = threads
                entry["evidence"] = [
                    "explicit fragment layout",
                    f"thread extent {threads}",
                    "packing lower bound assumes maximally packed 32-bit registers",
                    "index extent may include holes; allocation alone does not prove liveness",
                ]
                modeled[buffer] = entry
        elif buffer.scope() in ("local", "local.var") and size is not None:
            entry["modeled_registers_per_thread"] = {
                "lower": ceil(size * dtype.bits * dtype.lanes / 32),
                "upper": size * ceil(dtype.bits * dtype.lanes / 32),
            }
            entry["evidence"] = ["thread-private allocation; packing and storage elimination unresolved"]
            modeled[buffer] = entry
        elif buffer.scope() == "local.fragment" and size is not None:
            launch = [_int(v) for k, v in col.threads.items() if k.startswith("threadIdx")]
            if launch and all(x is not None for x in launch):
                entry["balanced_fp32_estimate"] = size / prod(launch) if str(buffer.dtype) == "float32" else None
                entry["evidence"] = ["conditional estimate: balanced layout across all launch threads; ownership not established"]
        storage.append(entry)

    return RegisterStorage(storage, modeled)
