"""CTA-dependent loop counts and launch timing, independently of kernel family."""

import heapq
from itertools import groupby
from math import prod
from tvm import tirx as tir
from tvm.arith import Analyzer


def collect_cta_work(col, loop, max_axis_points=4096):
    """Compress one varying block axis into runs; never expand the inner loop."""
    from .analysis import _int

    domains = col.block_domains
    result = {"precision": "unknown", "groups": [], "repetitions": None, "grid_blocks": None, "unknown": []}
    if loop is None or not domains:
        result["unknown"].append("no unique launch domain and pipeline loop")
        return result
    axes = sorted(domains)
    sizes = [_int(domains[axis][1].extent) for axis in axes]
    if any(n is None or n <= 0 for n in sizes):
        result["unknown"].append("symbolic CTA launch domain")
        return result
    result["grid_blocks"] = prod(sizes)
    extent = loop.extent
    constant = _int(extent)
    if constant is not None and constant >= 0:
        result.update(
            precision="exact",
            groups=[{"iterations": constant, "count": prod(sizes)}],
            repetitions=1,
            min_iterations=constant,
            max_iterations=constant,
            mean_iterations=float(constant),
        )
        return result
    variables = set()
    tir.stmt_functor.post_order_visit(extent, lambda n: variables.add(n) if isinstance(n, tir.Var) else None)
    varying = [i for i, axis in enumerate(axes) if domains[axis][0] in variables]
    if len(varying) != 1 or variables != {domains[axes[varying[0]]][0]}:
        result["unknown"].append("loop extent must depend on one resolved block axis")
        return result
    axis = varying[0]
    var, domain = domains[axes[axis]]
    start = _int(domain.min)
    if start is None or sizes[axis] > max_axis_points:
        result["unknown"].append("CTA work distribution exceeds bounded axis analysis")
        return result
    analyzer = Analyzer()
    counts = [
        _int(analyzer.simplify(tir.stmt_functor.substitute(extent, {var: tir.IntImm(var.dtype, start + i)}))) for i in range(sizes[axis])
    ]
    if any(n is None or n < 0 for n in counts):
        result["unknown"].append("unresolved per-CTA loop count")
        return result
    inner, outer = prod(sizes[:axis]), prod(sizes[axis + 1 :])
    groups = [{"iterations": n, "count": sum(1 for _ in items) * inner} for n, items in groupby(counts)]
    result.update(
        precision="exact",
        groups=groups,
        repetitions=outer,
        varying_axis=axes[axis],
        min_iterations=min(counts),
        max_iterations=max(counts),
        mean_iterations=sum(counts) / len(counts),
        assumptions=["CTA counts come from the original loop extent; x is the fastest launch axis"],
    )
    return result


def estimate_grid_cycles(distribution, timing_for_iterations, slots, uniform_waves):
    """Work-conserving dispatch estimate with exact IR work counts.

    For very large nonuniform grids, report a work-plus-tail approximation instead
    of enumerating every CTA. CUDA dispatch order and shared-SM interference are
    assumptions; the work distribution itself is independent of those assumptions.
    """
    if distribution.get("precision") != "exact" or not slots:
        return None
    groups, repetitions = distribution["groups"], distribution["repetitions"]
    timings = {n: timing_for_iterations(n) for n in {g["iterations"] for g in groups}}
    if not timings or any(timing is None for timing in timings.values()):
        return None
    costs = {n: timing["cycles"] for n, timing in timings.items()}
    work = sum(costs[g["iterations"]] * g["count"] for g in groups) * repetitions
    maximum, minimum = max(costs.values()), min(costs.values())
    if len(costs) == 1:
        cycles, method = maximum * uniform_waves, "uniform CTA waves"
    elif distribution["grid_blocks"] <= 262144:
        available = [0.0] * slots
        for _ in range(repetitions):
            for group in groups:
                cost = costs[group["iterations"]]
                for _ in range(group["count"]):
                    heapq.heapreplace(available, available[0] + cost)
        cycles, method = max(available), "CTA dispatch in launch order"
    else:
        cycles, method = max(maximum, work / slots + maximum * (1 - 1 / slots)), "work plus conservative one-CTA tail"
    return dict(
        cycles=cycles,
        method=method,
        work_cycles=work,
        min_cta_cycles=minimum,
        max_cta_cycles=maximum,
        work_lower_bound_cycles=max(maximum, work / slots),
        work_distribution=distribution,
        assumptions=[
            "equal resident slots; effective per-CTA rates share the SM throughput",
            "CUDA dispatch and resource contention are modeled, not observed",
        ],
    )
