"""Capture logical memory work from operation regions and loop dependencies."""

from math import prod

from .src.ir_utils import _int, loop_visits


def memory_unknowns(col):
    """Keep only uncertainty that may conceal a global-memory effect."""
    return list(getattr(col, "memory_unknown", col.unknown))


def cyclic_buffer_depth(col):
    """Infer explicit ring-buffer depth from shared-region indices.

    Manually pipelined kernels need not carry ``num_stages`` loop annotations.
    Their PrimFunc still exposes the version count when an access indexes a
    shared-buffer axis as ``phase % shape[axis]``.
    """
    from tvm import tirx as tir

    depths = []
    for op in col.operations:
        regions = op.reads + op.writes
        if op.kind not in ("async_copy", "tma_copy") or not any(region.buffer.scope() == "global" for region in regions):
            continue
        for region in regions:
            if not region.buffer.scope().startswith("shared"):
                continue
            for axis, interval in enumerate(region.ranges):
                if axis != 0 or axis >= len(region.buffer.shape):
                    continue
                extent = _int(region.buffer.shape[axis])
                if extent is None or extent <= 1:
                    continue
                moduli = []
                tir.stmt_functor.post_order_visit(
                    interval.min,
                    lambda node: moduli.append(_int(node.b)) if isinstance(node, tir.FloorMod) else None,
                )
                if extent in moduli:
                    depths.append(extent)
    return max([1, *depths])


def analyze_memory_accesses(col, buffer_facts):
    """Count requested accesses before clipping away masks or partial tiles.

    A partial final tile keeps its requested extent instead of independently
    bounding its start and end, which can inflate a small tail to a full tensor
    dimension. Scalar operations retain their enclosing loop visits. This is a
    logical-work ledger, not a cache or memory-transaction model.
    """
    accesses = []
    for op in col.operations:
        external = [
            (direction, region)
            for direction in ("reads", "writes")
            for region in getattr(op, direction)
            if region.buffer.scope() == "global"
        ]
        if not external:
            continue
        visits = loop_visits(op.loops)
        for direction, region in external:
            extents = [_int(axis.extent) for axis in region.ranges]
            elements = prod(extents) if all(value is not None and value >= 0 for value in extents) else None
            dtype = buffer_facts[region.buffer].dtype
            accesses.append(
                {
                    "operation": op.index,
                    "direction": direction,
                    "buffer": region.buffer.name,
                    "buffer_id": str(hash(region.buffer)),
                    "bytes": (elements * dtype.bits * dtype.lanes + 7) // 8 if elements is not None else None,
                    "visits": visits["max"],
                    "visit_precision": visits["precision"],
                    "predicated": bool(op.predicates),
                }
            )
    stages = [stage for op in col.operations for stage in op.pipeline_stages if type(stage) is int and stage > 0]
    unresolved_stages = any(stage is None for op in col.operations for stage in op.pipeline_stages)
    pipeline_depth = max([1, *stages, cyclic_buffer_depth(col)])
    launches = {tuple(sorted((key, str(value)) for key, value in op.launch_threads.items())) for op in col.operations}
    extents = [_int(value) for key, value in col.threads.items() if key.startswith("blockIdx.")]
    grid = prod(extents) if len(launches) == 1 and extents and all(value is not None and value > 0 for value in extents) else None
    return {
        "accesses": accesses,
        "grid_blocks": grid,
        "pipeline_depth": pipeline_depth,
        "pipeline_depth_precision": "lower_bound" if unresolved_stages else "exact",
        "dependencies": [{"operation": op.index, "predecessors": op.dependencies} for op in col.operations],
        "unknown": memory_unknowns(col),
    }
