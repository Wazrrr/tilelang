"""Conservative live register tiles, shared by every kernel family.

The collector supplies actual buffers and read/write order. Family phase labels
make the report readable; loop-carried storage is discovered from IR uses. This
estimate informs demand and ranking, but cannot establish a rejection proof.
"""

from math import prod


def analyze_live_tiles(col, buffer_facts, *, loop):
    """Conservative tile intervals, including attention's loop-carried state.

    These are logical demand estimates, never additional rejection evidence. Packing,
    automatic-layout replication, scalarization and storage reuse are unresolved.
    """
    from .src.ir_utils import _int, _exclusive
    from .src.ir_utils import in_loop

    thread_dims = [_int(v) for k, v in col.threads.items() if k.startswith("threadIdx.")]
    threads = prod(thread_dims) if thread_dims and all(thread_dims) else None
    loop_ops = [op for op in col.operations if in_loop(op, loop)]
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
        touches = [op for op in loop_ops if any(r.buffer.same_as(buffer) for r in op.reads + op.writes)]
        is_carried = bool(
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
            if (before and after) or (is_carried and in_loop(op, loop)):
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
        phases.append(
            {
                "operation": op.index,
                "buffers": active,
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
