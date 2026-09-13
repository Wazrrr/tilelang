"""Symbolic region geometry, independent of resource and timing models."""

from tvm import tirx as tir
from tvm.arith import Analyzer
from tvm.ir import Range
from .ir import Region


def _contains(outer, inner):
    if not outer.buffer.same_as(inner.buffer) or len(outer.ranges) != len(inner.ranges):
        return False
    ana = Analyzer()
    return all(
        ana.can_prove(a.min <= b.min) and ana.can_prove(a.min + a.extent >= b.min + b.extent) for a, b in zip(outer.ranges, inner.ranges)
    )


def _intersect(a, b):
    if not a.buffer.same_as(b.buffer) or len(a.ranges) != len(b.ranges):
        return None
    ana = Analyzer()
    ranges = []
    for x, y in zip(a.ranges, b.ranges):
        lo = ana.simplify(tir.max(x.min, y.min))
        hi = ana.simplify(tir.min(x.min + x.extent, y.min + y.extent))
        if ana.can_prove(hi <= lo):
            return None
        ranges.append(Range.from_min_extent(lo, ana.simplify(tir.max(0, hi - lo))))
    return Region(a.buffer, ranges, "conservative" if "conservative" in (a.precision, b.precision) else "exact")


def _subtract(region, written):
    overlap = _intersect(region, written)
    if overlap is None:
        return [region]
    if _contains(written, region):
        return []
    ana = Analyzer()
    core = list(region.ranges)
    pieces = []
    for i, (source, cut) in enumerate(zip(region.ranges, overlap.ranges)):
        left = ana.simplify(cut.min - source.min)
        right = ana.simplify(source.min + source.extent - cut.min - cut.extent)
        if not ana.can_prove(left >= 0) or not ana.can_prove(right >= 0):
            return [Region(region.buffer, region.ranges, "conservative")]
        for start, extent in ((source.min, left), (cut.min + cut.extent, right)):
            if not ana.can_prove(extent == 0):
                part = list(core)
                part[i] = Range.from_min_extent(ana.simplify(start), extent)
                pieces.append(Region(region.buffer, part, region.precision))
        core[i] = cut
    return pieces


def _bound(region, domains):
    if not domains:
        return region
    ana = Analyzer()
    ranges = []
    for r in region.ranges:
        low = ana.int_set(r.min, domains)
        high = ana.int_set(r.min + r.extent - 1, domains)
        if low.is_everything() or high.is_everything():
            return Region(region.buffer, region.ranges, "unknown")
        ranges.append(Range.from_min_extent(ana.simplify(low.min_value), ana.simplify(high.max_value - low.min_value + 1)))
    return Region(region.buffer, ranges, "conservative")


def _clip_global(region, loops):
    ana = Analyzer()
    for var, dom, _ in loops:
        ana.bind(var, dom)
    ranges = []
    precision = region.precision
    for r, shape in zip(region.ranges, region.buffer.shape):
        if ana.can_prove(r.min >= 0) and ana.can_prove(r.min + r.extent <= shape):
            ranges.append(r)
        else:
            low = ana.simplify(tir.max(0, r.min))
            high = ana.simplify(tir.min(shape, r.min + r.extent))
            ranges.append(Range.from_min_extent(low, ana.simplify(tir.max(0, high - low))))
            precision = "conservative"
    return Region(region.buffer, ranges, precision)
