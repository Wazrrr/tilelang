"""Conservative live register tiles, shared by every kernel family.

The collector supplies actual buffers and read/write order. Family phase labels
make the report readable; loop-carried storage is discovered from IR uses. This
estimate informs demand and ranking, but cannot establish a rejection proof.
"""

from math import prod


def _streamed_cast_storage(op, col, facts, active, threads):
    """A last-use, same-owner narrowing copy can retire source words as it casts.

    This refines the estimate only; it is never an allocation or spill proof.
    Retain four temporary words per owner for vectorized conversion/store code.
    """
    if op.kind != "copy" or len(op.reads) != 1 or len(op.writes) != 1 or not threads:
        return None
    src, dst = op.reads[0].buffer, op.writes[0].buffer
    a, b = facts[src], facts[dst]
    if a.scope != "local.fragment" or b.scope != a.scope or a.shape != b.shape or a.dtype.bits <= b.dtype.bits:
        return None
    entries = {e["buffer_id"]: e for e in active}
    if any(str(hash(buf)) not in entries or entries[str(hash(buf))]["loop_carried"] for buf in (src, dst)):
        return None
    if any(other.index > op.index and any(r.buffer.same_as(src) for r in other.reads) for other in col.operations):
        return None
    if any(other.index < op.index and any(r.buffer.same_as(dst) for r in other.writes) for other in col.operations):
        return None
    if a.layout is None or b.layout is None or a.replication != b.replication:
        return None
    from tvm import tirx as tir
    from tvm.arith import Analyzer
    from tvm.ir import Range
    from .ownership import _mapping
    from .src.ir_utils import _int

    # Only whole-buffer conversions qualify. Partial or shuffled copies can
    # retain source words and are deliberately left at the conservative sum.
    for region, shape in ((op.reads[0], a.shape), (op.writes[0], b.shape)):
        if any(_int(r.min) != 0 or _int(r.extent) != n for r, n in zip(region.ranges, shape)):
            return None
    indices = [tir.Var(f"cast_axis_{i}", "int32") for i in range(len(a.shape))]
    ana = Analyzer()
    for var, extent in zip(indices, a.shape):
        ana.bind(var, Range.from_min_extent(0, extent))
    source, ra = _mapping(a.layout, indices)
    destination, rb = _mapping(b.layout, indices)
    if (ra is None) != (rb is None):
        return None
    if ra is not None:
        ana.bind(ra, Range.from_min_extent(0, a.replication))
        destination = tir.stmt_functor.substitute(destination, {rb: ra})
    if not ana.can_prove(source == destination):
        return None
    bits = [entries[str(hash(buf))]["logical_bits_with_modeled_replication"] for buf in (src, dst)]
    if any(value is None for value in bits):
        return None
    saved = max(0, min(bits) - threads * 4 * 32)
    return dict(
        source=src.name,
        destination=dst.name,
        retired_bits_estimate=saved,
        basis="whole-buffer last-use narrowing copy with compiler-verified equal thread ownership; four scratch words per thread",
    )


def analyze_live_tiles(col, buffer_facts, *, loop):
    """Conservative tile intervals, including attention's loop-carried state.

    These are logical demand estimates, never additional rejection evidence. Packing,
    automatic-layout replication, scalarization and storage reuse are unresolved.
    """
    from .src.ir_utils import _int, _exclusive
    from .src.ir_utils import in_loop

    thread_dims = [_int(v) for k, v in col.threads.items() if k.startswith("threadIdx.")]
    threads = prod(thread_dims) if thread_dims and all(thread_dims) else None
    loops = col.serial_loops + col.pipeline_loops
    loop_ops = [[op for op in col.operations if in_loop(op, item)] for item in loops]
    carried = []
    entries = []
    for buffer in col.buffers:
        if buffer.scope() not in ("local.fragment", "local", "local.var"):
            continue
        facts = buffer_facts[buffer]
        bits = facts.logical_bits
        private = facts.scope != "local.fragment"
        replication = facts.replication
        if bits is not None:
            bits = bits * threads if private and threads else bits * replication if replication else None
        is_carried = False
        for operations in loop_ops:
            touches = [op for op in operations if any(r.buffer.same_as(buffer) for r in op.reads + op.writes)]
            is_carried |= bool(
                touches
                and any(r.buffer.same_as(buffer) for r in touches[0].reads)
                and any(any(r.buffer.same_as(buffer) for r in op.writes) for op in touches)
            )
        if is_carried:
            carried.append(buffer)
        entries.append((buffer, bits, is_carried))
    phases = []
    for op in col.operations:
        active = []
        for buffer, bits, is_carried in entries:
            before = any(
                old.index <= op.index and not _exclusive(old, op) and any(r.buffer.same_as(buffer) for r in old.writes)
                for old in col.operations
            )
            after = any(
                old.index >= op.index and not _exclusive(old, op) and any(r.buffer.same_as(buffer) for r in old.reads)
                for old in col.operations
            )
            if (before and after) or (is_carried and any(in_loop(op, item) for item in loops)):
                active.append(
                    {
                        "buffer": buffer.name,
                        "buffer_id": str(hash(buffer)),
                        "dtype": str(buffer.dtype),
                        "logical_bits_with_modeled_replication": bits,
                        "loop_carried": is_carried,
                    }
                )
        values = [entry["logical_bits_with_modeled_replication"] for entry in active]
        bits = sum(values) if all(value is not None for value in values) else None
        reuse = _streamed_cast_storage(op, col, buffer_facts, active, threads)
        if reuse is not None and bits is not None:
            bits -= reuse["retired_bits_estimate"]
        phases.append(
            {
                "operation": op.index,
                "buffers": active,
                "streamed_cast_storage": reuse,
                "packed_registers_per_block_estimate": (bits + 31) // 32 if bits is not None else None,
                "balanced_registers_per_thread_estimate": (bits + 32 * threads - 1) // (32 * threads)
                if bits is not None and threads
                else None,
            }
        )
    estimates = [phase["packed_registers_per_block_estimate"] for phase in phases]
    peak = max(estimates, default=0) if all(value is not None for value in estimates) else None
    return {
        "precision": "conservative" if peak is not None else "unknown",
        "phases": phases,
        "peak_registers_per_block_estimate": peak,
        "computing_threads_estimate": threads,
        "loop_carried_buffers": [b.name for b in carried],
        "assumptions": [
            "simultaneous tile intervals include possible storage reuse and mutually compatible paths",
            "automatic fragment replication and compiler temporaries are unmodeled",
            "loop-carried state persists; pipeline stages do not replicate accumulators",
            "liveness estimates do not strengthen the proven rejection bound",
        ],
    }
