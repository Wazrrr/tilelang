"""Dtype-aware tile pressure and conservative register liveness."""

from math import ceil, prod
from tilelang import tvm
from tvm.ir import Range
from .budget import resolve_register_budget


def analyze_live_tiles(col, specialization):
    """Conservative tile intervals, including attention's loop-carried state.

    These are logical demand estimates, never additional rejection evidence. Packing,
    automatic-layout replication, scalarization and storage reuse are unresolved.
    """
    from .analysis import _int, _exclusive
    from .specializations import in_loop

    thread_dims = [_int(v) for k, v in col.threads.items() if k.startswith("threadIdx.")]
    threads = prod(thread_dims) if thread_dims and all(thread_dims) else None
    loop_ops = [op for op in col.operations if in_loop(op, specialization.loop)]
    carried = []
    entries = []
    for buffer in col.buffers:
        if buffer.scope() not in ("local.fragment", "local", "local.var"):
            continue
        shape = [_int(x) for x in buffer.shape]
        dtype = tvm.DataType(buffer.dtype)
        bits = prod(shape) * dtype.bits * dtype.lanes if all(x is not None for x in shape) else None
        private = buffer.scope() != "local.fragment"
        layout = col.layouts.get(buffer.data)
        replication = _int(layout.replicate_size) if layout is not None and hasattr(layout, "replicate_size") else 1
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
            if (before and after) or (is_carried and in_loop(op, specialization.loop)):
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
                "phase": specialization.phase(op),
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


def analyze_register_pressure(col, config, target=None):
    from .analysis import Region, _int, _contains, _exclusive

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

    # Only full tile reads at a single operation establish simultaneous demand.
    # Separate operations/branches are never summed, nor are pipeline iterations.
    proven = 0
    block_registers = 0
    evidence = []
    for op in col.operations:
        if op.unknown or op.predicates or col.unknown or not op.demands:
            continue
        # A scalar copy/reduction can stream or fuse away its tile. Only dense
        # MMA accumulator state is required as a tile by a known instruction
        # family in this first physical model. Other values remain estimates.
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
            logical_size = prod(shape)
            layout = col.layouts.get(buffer.data)
            if layout is None:
                launch = [_int(v) for k, v in op.launch_threads.items() if k.startswith("threadIdx.")]
                if not launch or not all(x is not None and x > 0 for x in launch):
                    continue
                # All original launch threads are an upper bound on accumulator
                # owners. Warp specialization can dedicate fewer to compute;
                # added producer threads do not own this accumulator state.
                threads, replication = prod(launch), 1
                mapping_evidence = [
                    "automatic fragment layout: all launch threads bound accumulator ownership from above",
                    "replication ignored for the lower bound; no balanced mapping assumed",
                ]
            else:
                if buffer not in modeled:
                    continue
                threads = _int(layout.get_thread_size())
                output_size = prod(_int(x) for x in layout.get_output_shape())
                replication = _int(layout.replicate_size)
                # Equal volumes alone do not prove distinct physical storage.
                if not replication or output_size * threads != logical_size * replication:
                    continue
                try:
                    layout.inverse()
                except Exception:
                    continue
                mapping_evidence = ["explicit invertible fragment layout including replication"]
            dtype = tvm.DataType(buffer.dtype)
            bits = logical_size * replication * dtype.bits * dtype.lanes
            block_registers = max(block_registers, (bits + 31) // 32)
            # Integer ceiling preserves narrow packing and multi-register values.
            bound = (bits + 32 * threads - 1) // (32 * threads)
            if bound > proven:
                proven = bound
                evidence = [
                    f"operation {op.index}: full reads of dense MMA accumulator state",
                    *mapping_evidence,
                    f"{logical_size} elements × {dtype.bits} bits × {dtype.lanes} lanes × {replication} replication / (32 bits × {threads} threads), rounded up",
                    "maximum per-thread demand is at least the CTA average; maximally packed 32-bit registers",
                ]
    # Report conservative liveness of other register-resident intermediates.
    # These intervals summarize loops, retain possible partial overwrites, and
    # exclude mutually exclusive branch paths. They cannot justify rejection.
    live_sets = []
    for op in col.operations:
        active = []
        tile_upper = 0
        for buffer, entry in modeled.items():
            before = any(
                old.index <= op.index and not _exclusive(old, op) and any(r.buffer.same_as(buffer) for r in old.writes)
                for old in col.operations
            )
            after = any(
                old.index >= op.index and not _exclusive(old, op) and any(r.buffer.same_as(buffer) for r in old.reads)
                for old in col.operations
            )
            if before and after:
                active.append(buffer.name)
                tile_upper += entry["modeled_registers_per_thread"]["upper"]
        if active:
            live_sets.append(
                {"operation": op.index, "buffers": active, "precision": "conservative", "modeled_tile_registers_upper": tile_upper}
            )
    register_budget = resolve_register_budget(config, target)
    budget = register_budget["budget"]
    reject = budget is not None and proven > budget
    return {
        "logical_storage": storage,
        "live_tile_sets": live_sets,
        "modeled_lower_bound": proven if proven else None,
        "modeled_accumulator_registers_per_block": block_registers or None,
        "total_register_upper_bound": None,
        **register_budget,
        "evidence": evidence,
        "assumptions": [
            "compiler operand fragments and temporaries are unmodeled",
            "automatic fragment layout is unresolved; recognized warp-specialization policy is reported separately",
            "tile-state bounds are not bounds on total compiler registers",
        ],
        "decision": {
            "keep": not reject or config.mode == "report_only",
            "would_reject": reject,
            "status": "reject" if reject else "unknown",
        },
    }
