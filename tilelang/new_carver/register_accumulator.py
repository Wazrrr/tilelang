"""Proven register lower bounds for demanded dense MMA accumulators.

GEMM and attention share this operator-level proof. Accumulators are identified
by native operand identity, never by a kernel's buffer names. Allocation estimates
and conservative liveness do not strengthen the proof.
"""

from dataclasses import dataclass, field
from math import prod

from tilelang import tvm
from tvm.ir import Range


@dataclass
class AccumulatorBound:
    """Maximum established bounds across operations; zero means no proof."""

    registers_per_thread: int = 0
    registers_per_block: int = 0
    evidence: list[str] = field(default_factory=list)


def _required_accumulators(col):
    """Yield (operation, buffer, elements) for full reads and full demands.

    Scalar copies/reductions may stream or fuse away their tiles. Only the
    supported dense MMA accumulator state establishes simultaneous tile demand.
    """
    from .analysis import Region, _int, _contains

    for op in col.operations:
        if op.unknown or op.predicates or col.unknown or not op.demands:
            continue
        meta = op.metadata
        if meta is None or not hasattr(meta, "cRegion") or bool(getattr(meta, "isTcgen05", False)):
            continue
        seen = set()
        for read in op.reads:
            buffer = read.buffer
            if buffer in seen or buffer.scope() != "local.fragment" or not buffer.same_as(meta.c):
                continue
            seen.add(buffer)
            full = Region(buffer, [Range.from_min_extent(0, x) for x in buffer.shape])
            shape = [_int(x) for x in buffer.shape]
            if not all(x is not None and x > 0 for x in shape):
                continue
            if not _contains(read, full) or not any(_contains(demand, full) for demand in op.demands):
                continue
            yield op, buffer, prod(shape)


def _accumulator_ownership(col, op, buffer, logical_size, modeled_buffers):
    """Resolve (owner thread bound, replication, evidence), or leave unknown."""
    from .analysis import _int

    layout = col.layouts.get(buffer.data)
    if layout is None:
        launch = [_int(v) for k, v in op.launch_threads.items() if k.startswith("threadIdx.")]
        if not launch or not all(x is not None and x > 0 for x in launch):
            return None
        # Original launch threads bound the number of accumulator owners from
        # above. Added WS producer threads never enlarge this owner set.
        return (
            prod(launch),
            1,
            [
                "automatic fragment layout: all launch threads bound accumulator ownership from above",
                "replication ignored for the lower bound; no balanced mapping assumed",
            ],
        )

    if buffer not in modeled_buffers:
        return None
    threads = _int(layout.get_thread_size())
    output_size = prod(_int(x) for x in layout.get_output_shape())
    replication = _int(layout.replicate_size)
    # Equal volumes alone do not prove distinct physical storage.
    if not replication or output_size * threads != logical_size * replication:
        return None
    try:
        layout.inverse()
    except Exception:
        return None
    return threads, replication, ["explicit invertible fragment layout including replication"]


def analyze_accumulator_bound(col, modeled_buffers):
    """Establish dtype-aware bounds, taking a maximum across operations.

    Separate operations/branches are never summed, nor are pipeline iterations.
    The block and per-thread maxima are tracked independently.
    """
    result = AccumulatorBound()
    for op, buffer, logical_size in _required_accumulators(col):
        ownership = _accumulator_ownership(col, op, buffer, logical_size, modeled_buffers)
        if ownership is None:
            continue
        threads, replication, mapping_evidence = ownership
        dtype = tvm.DataType(buffer.dtype)
        bits = logical_size * replication * dtype.bits * dtype.lanes
        result.registers_per_block = max(result.registers_per_block, (bits + 31) // 32)
        # Integer ceiling preserves narrow packing and multi-register values.
        bound = (bits + 32 * threads - 1) // (32 * threads)
        if bound > result.registers_per_thread:
            result.registers_per_thread = bound
            result.evidence = [
                f"operation {op.index}: full reads of dense MMA accumulator state",
                *mapping_evidence,
                f"{logical_size} elements × {dtype.bits} bits × {dtype.lanes} lanes × {replication} replication / (32 bits × {threads} threads), rounded up",
                "maximum per-thread demand is at least the CTA average; maximally packed 32-bit registers",
            ]
    return result
