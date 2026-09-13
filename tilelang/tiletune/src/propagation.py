"""Backward tile demands and derived loop coverage on captured IR."""

from tilelang import tvm
from tvm import tirx as tir
from tvm.arith import Analyzer
from tvm.ir import Range
from .ir import Region, PropagationResult
from .ir_utils import _int, _domains
from .regions import _intersect, _subtract, _bound, _clip_global


def _store_axes(op):
    """Recognize dense separable stores; a bounding box may contain holes."""
    ana = Analyzer()
    axes, used = [], set()
    # ForKind.THREAD_BINDING (4) supplies launch coordinates, not tile axes.
    loop_vars = {v for v, _, kind in op.loops if kind != "4"}
    for index in op.metadata.indices:
        variables = set()
        tir.stmt_functor.post_order_visit(index, lambda n, variables=variables: variables.add(n) if isinstance(n, tir.Var) else None)
        varying = variables & loop_vars
        if not varying:
            axes.append(None)
            continue
        if len(varying) != 1:
            return None
        var = next(iter(varying))
        if var in used:
            return None
        offset = ana.simplify(index - var)
        remaining = set()
        tir.stmt_functor.post_order_visit(offset, lambda n, remaining=remaining: remaining.add(n) if isinstance(n, tir.Var) else None)
        if remaining & loop_vars:
            return None
        axes.append((var, offset))
        used.add(var)
    return axes


def _map_inputs(op, demand):
    meta = op.metadata
    ana = Analyzer()
    if op.kind in ("copy", "async_copy") and len(op.reads) == 1 and len(op.writes) == 1:
        src, dst = op.reads[0], op.writes[0]
        # Tensor slices often retain unit batch/head axes in global memory but
        # copy to a 2D shared tile. Match the non-unit axes without flattening
        # or losing the offsets of those batch/head axes.
        if len(src.ranges) != len(dst.ranges):
            src_axes = [i for i, r in enumerate(src.ranges) if not ana.can_prove_equal(r.extent, 1)]
            dst_axes = [i for i, r in enumerate(dst.ranges) if not ana.can_prove_equal(r.extent, 1)]
            if len(src_axes) == len(dst_axes) and all(
                ana.can_prove_equal(src.ranges[i].extent, dst.ranges[j].extent) for i, j in zip(src_axes, dst_axes)
            ):
                ranges = list(src.ranges)
                for i, j in zip(src_axes, dst_axes):
                    ranges[i] = Range.from_min_extent(
                        ana.simplify(src.ranges[i].min + demand.ranges[j].min - dst.ranges[j].min), demand.ranges[j].extent
                    )
                return [Region(src.buffer, ranges, demand.precision)]
        if len(src.ranges) == len(dst.ranges) and all(ana.can_prove_equal(a.extent, b.extent) for a, b in zip(src.ranges, dst.ranges)):
            return [
                Region(
                    src.buffer,
                    [
                        Range.from_min_extent(ana.simplify(s.min + q.min - d.min), q.extent)
                        for s, d, q in zip(src.ranges, dst.ranges, demand.ranges)
                    ],
                    demand.precision,
                )
            ]
    if meta is not None and hasattr(meta, "aRegion") and hasattr(meta, "transA") and not bool(getattr(meta, "isTcgen05", False)):
        c = Region.from_ir(meta.cRegion)
        if len(demand.ranges) == 2 and len(c.ranges) == 2:
            m, n = [Range.from_min_extent(ana.simplify(q.min - base.min), q.extent) for q, base in zip(demand.ranges, c.ranges)]
            result = []
            for original, axis, sub in ((meta.aRegion, 1 if meta.transA else 0, m), (meta.bRegion, 0 if meta.transB else 1, n)):
                region = Region.from_ir(original)
                if len(region.ranges) != 2:
                    return [Region(r.buffer, r.ranges, "conservative") for r in op.reads]
                base = region.ranges[axis]
                region.ranges[axis] = Range.from_min_extent(ana.simplify(base.min + sub.min), sub.extent)
                result.append(region)
            if _int(meta.clearAccum) != 1:
                result.append(demand)
            return result
    if meta is not None and hasattr(meta, "dim") and hasattr(meta, "srcRegion"):
        src = Region.from_ir(meta.srcRegion)
        dst = Region.from_ir(meta.dstRegion)
        dim = int(meta.dim)
        if len(src.ranges) == len(dst.ranges) + 1:
            j = 0
            for i in range(len(src.ranges)):
                if i != dim:
                    src.ranges[i] = Range.from_min_extent(
                        ana.simplify(src.ranges[i].min + demand.ranges[j].min - dst.ranges[j].min), demand.ranges[j].extent
                    )
                    j += 1
            return [src] + ([demand] if not meta.clear else [])
    if op.kind == "elementwise":
        domains = _domains(tuple(loop for loop in op.loops if loop[2] != "4"))
        axes = _store_axes(op)
        if axes is not None:
            loop_ranges = {v: r for v, r, _ in op.loops}
            for axis, requested in zip(axes, demand.ranges):
                if axis is not None:
                    var, offset = axis
                    dom = loop_ranges[var]
                    low = ana.simplify(tir.max(dom.min, requested.min - offset))
                    high = ana.simplify(tir.min(dom.min + dom.extent, requested.min + requested.extent - offset) - 1)
                    domains[var] = tvm.arith.IntervalSet(low, high)
        return [_bound(r, domains) for r in op.reads]
    return [Region(r.buffer, r.ranges, "conservative") for r in op.reads]


def _written_region(op, region):
    """Expand scalar tile axes while retaining symbolic launch coordinates.

    Native copy/GEMM/reduction operators already carry tile regions. Launch
    domains are used separately for ownership, boundary checks and grid cost.
    """
    if op.kind == "elementwise":
        return _bound(region, _domains(tuple(loop for loop in op.loops if loop[2] != "4")))
    return region


def _kernel_outputs(col):
    """Capture the kernel's global writes as propagation roots."""
    outputs = [_written_region(op, region) for op in col.operations for region in op.writes if region.buffer.scope() == "global"]
    if not outputs:
        raise ValueError("TileTune requires at least one captured global output write")
    return outputs


def _propagate_tiles(col, outputs):
    """Trace output tiles once, recording the demands used by every analysis.

    Regions retain symbolic launch/iteration offsets. Loop coverage is derived
    from these inputs; it does not require another backward traversal.
    """
    for op in col.operations:
        op.demands.clear()
    demands = list(outputs)
    origins = {id(region): () for region in demands}
    for op in reversed(col.operations):
        matched = []
        for write in op.writes:
            bounded_write = _written_region(op, write)
            for demand in list(demands):
                overlap = _intersect(bounded_write, demand)
                if overlap is not None:
                    op.demands.append(overlap)
                    matched.extend(_map_inputs(op, overlap))
                    # Subtract overwritten rectangles; conditional writes retain
                    # both reaching versions. Unknown overlap keeps a bound.
                    if not op.predicates and (op.kind != "elementwise" or _store_axes(op) is not None):
                        remaining = _subtract(demand, bounded_write)
                        for region in remaining:
                            origins[id(region)] = origins[id(demand)]
                        demands = [r for r in demands if r is not demand] + remaining
        for region in matched:
            demands.append(region)
            # Elementwise mapping already bounds its scalar axes, so do not
            # multiply their extents into the mapped tile a second time.
            origins[id(region)] = () if op.kind == "elementwise" else op.loops
    inputs, coverage, input_loops = [], [], []
    for region in demands:
        if region.buffer.scope() == "global":
            loops = origins[id(region)]
            region = _clip_global(region, loops)
            inputs.append(region)
            input_loops.append(loops)
            coverage.append(_bound(region, _domains(tuple(loop for loop in loops if loop[2] != "4"))))
    return PropagationResult(col.operations, inputs, coverage, col.unknown, input_loops)
