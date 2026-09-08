"""Opt-in, readable snapshots of intermediate analysis data.

Each checkpoint is serialized immediately, preserving its stage of the analysis.
One complete analysis block is appended on exit, including on analysis failure.
A lock keeps blocks from compilation threads together. Trace errors never change
resource decisions or ranking. No snapshots are constructed when tracing is off.
"""

import json
import logging
import os
from pathlib import Path
import sys
from threading import Lock
from uuid import uuid4


_WRITE_LOCK = Lock()
_LOG = logging.getLogger(__name__)


class AnalysisTrace:
    def __init__(self, path=None):
        self.path = path
        self.entries = []

    def __enter__(self):
        return self

    def record(self, name, snapshot):
        """Evaluate a lazy snapshot only when enabled; freeze it as readable text."""
        if self.path is None:
            return
        frame = sys._getframe(1)
        location = f"{Path(frame.f_code.co_filename).name}:{frame.f_lineno} in {frame.f_code.co_name}"
        del frame
        try:
            value = snapshot()
            body = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False, default=str)
        except Exception as error:
            body = json.dumps({"trace_error": f"{type(error).__name__}: {error}"})
            _LOG.warning("New Carver trace checkpoint %s failed: %s", name, error)
        self.entries.append(f"[{len(self.entries) + 1:02d}] {name} ({location})\n{body}\n")

    def __exit__(self, exc_type, error, traceback):
        if self.path is None:
            return False
        if error is not None:
            self.record("analysis_error", lambda: {"type": exc_type.__name__, "message": str(error)})
        block = (
            f"===== New Carver analysis {uuid4().hex[:12]} (pid={os.getpid()}) =====\n\n"
            + "\n".join(self.entries)
            + f"\n===== {'FAILED' if error is not None else 'COMPLETE'} =====\n\n"
        )
        try:
            path = Path(self.path)
            with _WRITE_LOCK:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(block)
        except Exception as trace_error:
            _LOG.warning("Cannot write New Carver trace to %s: %s", self.path, trace_error)
        return False


def _buffer(buffer):
    return {
        "type": "Buffer",
        "buffer": buffer.name,
        "buffer_id": str(hash(buffer)),
        "data_var": str(buffer.data),
        "scope": buffer.scope(),
        "dtype": str(buffer.dtype),
        "shape": [str(extent) for extent in buffer.shape],
    }


def _loops(loops):
    return [{"var": str(var), "min": str(domain.min), "extent": str(domain.extent), "kind": kind} for var, domain, kind in loops]


def _native_value(value):
    """Describe native operands/regions; symbolic expressions remain IR strings."""
    from tilelang import tvm
    import tvm_ffi
    from tvm import tirx as tir
    from .analysis import Region

    if isinstance(value, tir.Buffer):
        return _buffer(value)
    if isinstance(value, tir.BufferRegion):
        return {"type": "BufferRegion", **Region.from_ir(value).to_dict()}
    if isinstance(value, tvm.ir.Range):
        return {"min": str(value.min), "extent": str(value.extent)}
    if isinstance(value, list | tuple | tvm_ffi.Array):
        return [_native_value(item) for item in value]
    if value is None or isinstance(value, str | bool | int | float):
        return value
    return str(value)


def _operation(op):
    # Reflected fields used by GEMM, copy, reduction and scalar-store analysis.
    fields = (
        "a",
        "b",
        "c",
        "aRegion",
        "bRegion",
        "cRegion",
        "transA",
        "transB",
        "clearAccum",
        "m",
        "n",
        "k",
        "isTcgen05",
        "src",
        "dst",
        "src_range",
        "dst_range",
        "srcRegion",
        "dstRegion",
        "dim",
        "clear",
        "value",
        "indices",
    )
    return {
        "type": "Operation",
        **op.to_dict(),
        "metadata": {
            "type": type(op.metadata).__name__,
            "fields": {name: _native_value(getattr(op.metadata, name)) for name in fields if hasattr(op.metadata, name)},
        },
        "demands": [region.to_dict() for region in op.demands],
        "launch_threads": {name: str(extent) for name, extent in op.launch_threads.items()},
        "pipeline_stages": list(op.pipeline_stages),
    }


def collector_snapshot(col):
    def loop(node):
        return {
            "var": str(node.loop_var),
            "min": str(node.min),
            "extent": str(node.extent),
            "annotations": {str(key): str(value) for key, value in node.annotations.items()},
        }

    return {
        "type": "_Collector",
        "buffers": [_buffer(buffer) for buffer in col.buffers],
        "operations": [_operation(op) for op in col.operations],
        "threads": {name: str(extent) for name, extent in col.threads.items()},
        "block_domains": {
            name: {"var": str(var), "min": str(domain.min), "extent": str(domain.extent)}
            for name, (var, domain) in col.block_domains.items()
        },
        "layouts": {str(var): str(layout) for var, layout in col.layouts.items()},
        "pipeline_loops": [loop(node) for node in col.pipeline_loops],
        "serial_loops": [loop(node) for node in col.serial_loops],
        "bindings": {str(var): str(value) for var, value in col.bindings.items()},
        "unknown": list(col.unknown),
    }


def propagation_snapshot(result):
    return {
        "type": "PropagationResult",
        **result.to_dict(),
        "operation_demands": [{"operation": op.index, "demands": [r.to_dict() for r in op.demands]} for op in result.operations],
        "input_loops": [_loops(loops) for loops in result.input_loops],
    }
