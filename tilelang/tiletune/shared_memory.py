"""Staged shared allocations and storage reuse from buffer lifetimes."""

from math import prod
from .src.ir_utils import _int


def analyze_shared_memory(col, buffer_facts, pass_configs=None):
    def product(values):
        values = [_int(v) for v in values]
        return prod(values) if all(v is not None and v >= 0 for v in values) else None

    shared = []
    for buffer in col.buffers:
        if not buffer.scope().startswith("shared") or buffer.scope() == "shared.tmem":
            continue
        facts = buffer_facts[buffer]
        logical = (facts.logical_bits + 7) // 8 if all(x is not None and x >= 0 for x in facts.shape) else None
        factors = [1]
        for op in col.operations:
            if any(r.buffer.same_as(buffer) for r in op.writes):
                factors.append(product(max(1, s) if s is not None else None for s in op.pipeline_stages))
        stages = max(factors) if all(x is not None for x in factors) else None
        shared.append(
            {
                "buffer": buffer.name,
                "buffer_id": str(hash(buffer)),
                "dtype": str(buffer.dtype),
                "shape": [str(x) for x in buffer.shape],
                "logical_bytes": logical,
                "pipeline_copies_estimate": stages,
                "allocated_bytes_estimate": logical * stages if logical is not None and stages is not None else None,
            }
        )
    storage = shared_storage_plan(col, shared, pass_configs)
    return {
        "shared_allocations": shared,
        "shared_memory_bytes_estimate": storage["arena_bytes_estimate"],
        "shared_memory_allocated_sum_bytes": storage["allocated_sum_bytes"],
        "shared_storage_plan": storage,
        "assumptions": [
            "only shared buffers written in a pipeline receive stage copies; layout padding and compiler barriers are unmodeled",
            "shared tile lifetimes predict storage reuse across disjoint regions; entire repeated loops remain overlapping",
            "shared arena size is an estimate, not a compiler allocation guarantee or a rejection bound",
        ],
    }


def shared_storage_plan(col, allocations, pass_configs=None):
    sizes = [a["allocated_bytes_estimate"] for a in allocations]
    total = sum(sizes) if all(s is not None for s in sizes) else None
    result = dict(
        allocated_sum_bytes=total,
        arena_bytes_estimate=total,
        intervals=[],
        reuse_bytes_estimate=0,
        precision="estimate",
        method="sum of shared allocations",
        unknown=[],
    )
    if not allocations:
        return result
    if (pass_configs or {}).get("tl.disable_shared_memory_reuse", False):
        result["method"] = "shared-memory reuse disabled by pass configuration"
        return result
    if col.unknown or total is None:
        result.update(precision="unknown", unknown=["unresolved accesses, aliases or allocation sizes; reuse not predicted"])
        return result
    buffers = {str(hash(b)): b for b in col.buffers}
    # Treat every repeated serial/pipeline body as a simultaneous working set.
    # This also protects loop-carried state and async prefetch from naive
    # first/last-use reuse inside the lexical body.
    loop_ranges = {}
    for op in col.operations:
        for var, _, kind in op.loops:
            if kind in ("4", "1"):  # launch/thread binding and tile-parallel axes
                continue
            key = hash(var)
            lo, hi = loop_ranges.get(key, (op.index, op.index))
            loop_ranges[key] = min(lo, op.index), max(hi, op.index)
    intervals = []
    for order, alloc in enumerate(allocations):
        buffer = buffers[alloc["buffer_id"]]
        uses = [op for op in col.operations if any(r.buffer.same_as(buffer) for r in op.reads + op.writes)]
        if not uses:
            # An allocation without reflected uses may be visible to a later
            # pass; keep its storage live rather than dropping it.
            lo, hi = 0, len(col.operations)
        else:
            lo, hi = uses[0].index, uses[-1].index
            for op in uses:
                for var, _, kind in op.loops:
                    if kind not in ("4", "1"):
                        start, end = loop_ranges[hash(var)]
                        lo, hi = min(lo, start), max(hi, end)
        intervals.append(
            dict(
                buffer=alloc["buffer"],
                buffer_id=alloc["buffer_id"],
                start=lo,
                end=hi,
                bytes=alloc["allocated_bytes_estimate"],
                scope=buffer.scope(),
                order=order,
            )
        )
    # Stable linear scan with a best-fit free arena. Sharing is restricted to
    # identical storage scopes; static barriers and dynamic tiles remain apart.
    arenas = {}
    for scope in sorted({i["scope"] for i in intervals}):
        active, free, top = [], [], 0
        for item in sorted((i for i in intervals if i["scope"] == scope), key=lambda i: (i["start"], i["order"])):
            kept = []
            for old in active:
                if old["end"] < item["start"]:
                    free.append((old["offset_bytes_estimate"], old["bytes"]))
                else:
                    kept.append(old)
            active = kept
            merged = []
            for offset, size in sorted(free):
                if merged and merged[-1][0] + merged[-1][1] == offset:
                    merged[-1] = (merged[-1][0], merged[-1][1] + size)
                else:
                    merged.append((offset, size))
            free = merged
            # Sizes/offsets are logical bytes; layout padding and alignment are
            # deliberately not presented as compiler allocation guarantees.
            fits = [(size, offset, j) for j, (offset, size) in enumerate(free) if size >= item["bytes"]]
            if fits:
                size, offset, j = min(fits)
                free.pop(j)
                if size > item["bytes"]:
                    free.append((offset + item["bytes"], size - item["bytes"]))
            elif free and free[-1][0] + free[-1][1] == top:
                offset, _ = free.pop()
                top = offset + item["bytes"]
            else:
                offset = top
                top += item["bytes"]
            item["offset_bytes_estimate"] = offset
            active.append(item)
        arenas[scope] = top
    arena = sum(arenas.values())
    result.update(
        arena_bytes_estimate=arena,
        reuse_bytes_estimate=total - arena,
        intervals=intervals,
        scope_arena_bytes_estimate=arenas,
        method="tile lifetime arena; repeated loops expanded to full intervals",
    )
    return result
