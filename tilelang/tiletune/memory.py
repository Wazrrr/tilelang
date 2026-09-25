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
                    lambda node, moduli=moduli: moduli.append(_int(node.b)) if isinstance(node, tir.FloorMod) else None,
                )
                if extent in moduli:
                    depths.append(extent)
    return max([1, *depths])


def analyze_compute_intensity(col, buffer_facts, grid_blocks):
    """Count dynamic matrix FLOPs and distinct global bytes for a roofline gate.

    This deliberately adds no instruction schedule, pipeline recurrence, or
    measured rate. Dynamic FLOPs multiply per-iteration matrix work by the
    collected loop visits; tensor bytes count each global buffer once.
    """
    from .compute import operation_work

    unknown = []
    seen = {}
    for buffer in col.buffers:
        if buffer.scope() != "global":
            continue
        identity = str(buffer.data)
        if identity not in seen:
            seen[identity] = buffer_facts[buffer].logical_bits

    unique_bytes = 0
    for identity, bits in seen.items():
        if bits is None:
            unknown.append(f"unresolved global tensor size for {identity}")
        else:
            unique_bytes += bits // 8

    flops_per_cta = 0
    for op in col.operations:
        flops = operation_work(op, col).get("gemm_flops")
        visits = loop_visits(op.loops)["max"]
        if flops is None or visits is None:
            unknown.append(f"unresolved matrix work for operation {op.index}")
        elif flops:
            flops_per_cta += flops * visits
    compute_work = flops_per_cta * grid_blocks if grid_blocks is not None else None
    if grid_blocks is None:
        unknown.append("unresolved grid size")
    return {
        "compute_work": compute_work if not unknown else None,
        "unique_global_bytes": unique_bytes if not unknown else None,
        "matrix_flops_per_cta": flops_per_cta,
        "grid_blocks": grid_blocks,
        "precision": "unknown" if unknown else "estimate",
        "unknown": unknown,
        "assumptions": [
            "roofline split only; no instruction schedule, cache model or measured rate",
            "dynamic matrix work multiplies per-iteration FLOPs by the collected loop visits",
            "unique bytes count distinct global buffers, not repeated logical traffic",
        ],
    }


def resident_warps_estimate(col, shared_bytes, device_limits):
    """Estimate resident warps from shared memory and launch limits.

    This coarse latency-hiding proxy intentionally omits compiler register
    allocation so the bound-aware ordering remains profile-free.
    """
    limits = device_limits or {}
    if shared_bytes is None or not limits:
        return None
    warp = limits.get("warp_size", 32)
    threads = 1
    for key, value in col.threads.items():
        if key.startswith("threadIdx."):
            extent = _int(value)
            if extent is None or extent <= 0:
                return None
            threads *= extent
    if not threads:
        return None
    bounds = []
    if limits.get("shared_memory_per_sm"):
        bounds.append(limits["shared_memory_per_sm"] // max(shared_bytes, 1))
    if limits.get("max_threads_per_sm"):
        bounds.append(limits["max_threads_per_sm"] // threads)
    if limits.get("max_blocks_per_sm"):
        bounds.append(limits["max_blocks_per_sm"])
    if not bounds:
        return None
    resident = max(0, min(bounds))
    warps = threads // max(warp, 1)
    return {
        "resident_blocks_per_sm_estimate": resident,
        "warps_per_block": warps,
        "active_warps_per_sm_estimate": resident * warps,
    }


def analyze_memory_accesses(col, buffer_facts, *, include_dependencies=True):
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
        "dependencies": (
            [{"operation": op.index, "predecessors": op.dependencies} for op in col.operations]
            if include_dependencies
            else None
        ),
        "dependency_precision": "exact" if include_dependencies else "disabled",
        "unknown": memory_unknowns(col),
    }
