"""Bounded region schedules for serial work and independent software pipelines.

Domains are split only when work changes (causal bounds, branches or tile tails).
Uniform intervals stay symbolic; no launch-sized list or loop unrolling is used.
The bounded splitter fails explicitly for unresolved data-dependent control flow.
"""

from dataclasses import replace
from itertools import groupby
from math import prod

from tilelang import tvm
from tvm import tirx as tir
from tvm.arith import Analyzer
from tvm.ir import Range

from .compute import operation_work
from .src.structural_key import StructuralKey
from .src.ir_utils import _int, rectangular_scalar_access
from .src.regions import _bound, _clip_global


from tiletune_core.region_schedule import UnresolvedRegion as UnresolvedRegion


def _analyzer(domains):
    ana = Analyzer()
    for var, (lo, extent) in domains.items():
        ana.bind(var, Range.from_min_extent(lo, extent))
    return ana


def _constant(expr, domains, code="unresolved_memory_bounds"):
    result = _int(expr)
    if result is None:
        bound = Analyzer().int_set(expr, {v: tvm.arith.IntervalSet(lo, lo + n - 1) for v, (lo, n) in domains.items()})
        lo, hi = _int(bound.min_value), _int(bound.max_value)
        if lo is not None and lo == hi:
            return lo
        raise UnresolvedRegion(code, "expression varies within the region domain")
    return result


def _split(domain, evaluate, budget):
    """Return ordered (length, value) runs, bisecting only nonuniform intervals."""
    var, lo, count = domain
    if not count:
        return []
    budget[0] -= 1
    if budget[0] < 0:
        raise UnresolvedRegion("unsupported_scheduling", "bounded region analysis exceeded 4096 partitions")
    try:
        return [(count, evaluate(var, lo, count))]
    except UnresolvedRegion:
        if count <= 1:
            raise
    half = count // 2
    runs = _split((var, lo, half), evaluate, budget) + _split((var, lo + half, count - half), evaluate, budget)
    return [(sum(n for n, _ in group), value) for value, group in groupby(runs, key=lambda pair: pair[1])]


def region_tree(col):
    """Lexical loop identity retains sequential siblings and nested serial work."""
    loops = {loop.loop_var: loop for loop in col.serial_loops + col.pipeline_loops}
    root = []
    for op in col.operations:
        children = root
        for var, _, kind in op.loops:
            if kind in ("1", "4"):
                continue
            if var not in loops:
                raise UnresolvedRegion("unsupported_scheduling", "unrecognized serial loop kind or annotation")
            if not children or children[-1].get("var") != var:
                children.append(dict(var=var, loop=loops[var], children=[]))
            children = children[-1]["children"]
        children.append(dict(operation=op.index))
    return root


def _operation(op, phase, domains, col):
    external = phase.get("external_work")
    if (
        not op.predicates
        and external is not None
        and all(value is not None for value in external.values())
        and not (external["read_groups"] and not external["read_bytes"])
        and all(value is not None for value in phase["work"].values())
        and all(_int(r.extent) is not None for _, r, kind in op.loops if kind == "1")
    ):
        # The original full-domain analysis already proved constant work. A
        # subregion cannot change it; reuse these immutable counts directly.
        return dict(operation=op.index, work=phase["work"], external_work=external, active=True)
    parallel = {v: (_constant(r.min, domains), _constant(r.extent, domains)) for v, r, kind in op.loops if kind == "1"}
    active = dict(parallel)
    # Rectangular scalar tail guards can shrink tile axes. Nonrectangular or
    # data-dependent predicates remain unsupported; collectives cannot be masked.
    for predicate in op.predicates:
        ana = _analyzer({**domains, **active})
        if ana.can_prove(predicate):
            continue
        if ana.can_prove(tir.Not(predicate)):
            return dict(
                operation=op.index,
                work={k: 0 for k in phase["work"]},
                external_work=dict(read_bytes=0, write_bytes=0, read_groups=0),
                active=False,
            )
        if op.kind != "elementwise":
            raise UnresolvedRegion("unsupported_scheduling", "conditional collective or tile operation is not uniform")
        conditions = []

        def flatten(expr, conditions=conditions):
            if isinstance(expr, tir.And):
                flatten(expr.a)
                flatten(expr.b)
            else:
                conditions.append(expr)

        flatten(predicate)
        for condition in conditions:
            ana = _analyzer({**domains, **active})
            if ana.can_prove(condition):
                continue
            candidates = []
            for var, (lo, n) in active.items():
                others = {v: bounds for v, bounds in {**domains, **active}.items() if v != var}
                try:

                    def truth(_, start, length, others=others, var=var, condition=condition):
                        check = _analyzer({**others, var: (start, length)})
                        if check.can_prove(condition):
                            return True
                        if check.can_prove(tir.Not(condition)):
                            return False
                        raise UnresolvedRegion("unresolved_memory_bounds", "unresolved scalar guard")

                    runs = _split((var, lo, n), truth, [4096])
                    position, kept = lo, []
                    for length, enabled in runs:
                        if enabled:
                            kept.append((position, length))
                        position += length
                    if len(kept) <= 1:
                        candidates.append((var, kept[0] if kept else (lo, 0)))
                except UnresolvedRegion:
                    pass
            if not candidates:
                raise UnresolvedRegion("unsupported_scheduling", "unresolved or nonrectangular scalar guard")
            var, bounds = candidates[0]
            active[var] = bounds
    if any(n == 0 for _, n in active.values()):
        return dict(
            operation=op.index,
            work={k: 0 for k in phase["work"]},
            external_work=dict(read_bytes=0, write_bytes=0, read_groups=0),
            active=False,
        )
    work = phase["work"]
    if active != parallel:
        loops = tuple((v, Range.from_min_extent(*active[v]), kind) if v in active else (v, r, kind) for v, r, kind in op.loops)
        work = operation_work(replace(op, loops=loops), col)
    external = {}
    for name, regions in (("read", op.reads), ("write", op.writes)):
        amounts, seen = [], set()
        for region in regions:
            if region.buffer.scope() != "global":
                continue
            # Resolve singleton CTA coordinates before bounding parallel axes.
            # Otherwise an interval analyzer may union unrelated arms of a
            # metadata lookup and turn one row tile into a span across groups.
            fixed = {v: tir.const(lo, v.dtype) for v, (lo, n) in domains.items() if n == 1}
            if fixed:
                from .src.ir import Region

                ana = Analyzer()
                region = Region(
                    region.buffer,
                    [
                        Range.from_min_extent(
                            ana.simplify(tir.stmt_functor.substitute(r.min, fixed)),
                            ana.simplify(tir.stmt_functor.substitute(r.extent, fixed)),
                        )
                        for r in region.ranges
                    ],
                    region.precision,
                )
            intervals = {v: tvm.arith.IntervalSet(lo, lo + n - 1) for v, (lo, n) in active.items()}
            if op.kind == "elementwise" and not rectangular_scalar_access(region, op.loops):
                raise UnresolvedRegion("unresolved_memory_bounds", "nonrectangular scalar access requires an exact address-count model")
            bounded = _bound(region, intervals)
            loops = tuple((v, Range.from_min_extent(lo, n), "4") for v, (lo, n) in domains.items())
            bounded = _clip_global(bounded, loops)
            key = (region.buffer, tuple((StructuralKey(r.min), StructuralKey(r.extent)) for r in bounded.ranges))
            if key in seen:
                continue
            seen.add(key)
            dims = [_constant(r.extent, domains) for r in bounded.ranges]
            if any(n < 0 for n in dims):
                raise UnresolvedRegion("unresolved_memory_bounds", "negative external region extent")
            dtype = tvm.DataType(region.buffer.dtype)
            amounts.append((prod(dims) * dtype.bits * dtype.lanes + 7) // 8)
        external[name + "_bytes"] = sum(amounts)
        if name == "read":
            external["read_groups"] = sum(n > 0 for n in amounts)
    return dict(operation=op.index, work=work, external_work=external, active=True)


def build_region_schedule(col, phases, pressure):
    """Build compressed CTA classes, each containing a serial region schedule."""
    tree = region_tree(col)
    phase_map = {p["operation"]: p for p in phases}
    plans = getattr(col, "ampere_plans", {})
    liveness = {p["operation"]: p["buffers"] for p in pressure["tile_liveness"]["phases"]}
    budget = [4096]

    def sequence(nodes, domains):
        result = []
        for node in nodes:
            if "operation" in node:
                op = col.operations[node["operation"]]
                result.append(_operation(op, phase_map[op.index], domains, col))
                continue
            loop, var = node["loop"], node["var"]
            lo = _constant(loop.min, domains, "unsupported_scheduling")
            n = _constant(loop.extent, domains, "unsupported_scheduling")
            if n < 0:
                raise UnresolvedRegion("unsupported_scheduling", "negative serial loop extent")
            depth = _int(loop.annotations.get("num_stages", 0))
            if depth and (var not in plans or plans[var]["status"] != "predicted"):
                raise UnresolvedRegion("unsupported_scheduling", "independent pipeline has no verified compiler plan")
            if depth and any("loop" in child for child in node["children"]):
                raise UnresolvedRegion("unsupported_scheduling", "nested software pipelines are not modeled")
            runs = _split(
                (var, lo, n),
                lambda v, start, length, children=node["children"], domains=domains: sequence(children, {**domains, v: (start, length)}),
                budget,
            )
            ids = [p["operation"] for p in node["children"] if "operation" in p]
            result.append(
                dict(
                    kind="pipeline" if depth else "serial",
                    iterations=n,
                    runs=[dict(count=count, body=body) for count, body in runs],
                    plan=plans.get(var),
                    depth=depth,
                    operations=ids,
                    dependencies=sorted({d for i in ids for d in col.operations[i].dependencies if d not in ids}),
                    live_buffers=sorted({b["buffer"] for i in ids for b in liveness[i]}),
                )
            )
        return result

    axes = sorted(col.block_domains)
    if not axes:
        raise UnresolvedRegion("unsupported_scheduling", "no resolved CTA launch domain")
    domains = {v: (_int(r.min), _int(r.extent)) for v, r in col.block_domains.values()}
    if any(lo is None or n is None or n <= 0 for lo, n in domains.values()):
        raise UnresolvedRegion("unsupported_scheduling", "symbolic CTA launch domain")
    # First try the entire grid. Otherwise split one varying axis and compress
    # the other launch axes as repetitions in CUDA's x-fastest order.
    try:
        schedules = [(prod(n for _, n in domains.values()), sequence(tree, domains))]
        repetitions = 1
    except UnresolvedRegion:
        for axis_index, axis in enumerate(axes):
            var, _ = col.block_domains[axis]
            lo, n = domains[var]
            try:
                runs = _split((var, lo, n), lambda v, start, length: sequence(tree, {**domains, v: (start, length)}), budget)
                inner = prod(domains[col.block_domains[a][0]][1] for a in axes[:axis_index])
                repetitions = prod(domains[col.block_domains[a][0]][1] for a in axes[axis_index + 1 :])
                schedules = [(count * inner, body) for count, body in runs]
                break
            except UnresolvedRegion:
                continue
        else:
            # A tail or metadata branch can vary on more than one launch axis.
            # Keep identical rows compressed while splitting the outer axes,
            # then retain CUDA's x-fastest order in the bounded group list.
            def grid_rows(axis_index, bounds):
                if axis_index < 0:
                    return [(1, sequence(tree, bounds))]
                var, _ = col.block_domains[axes[axis_index]]
                lo, n = bounds[var]
                runs = _split(
                    (var, lo, n),
                    lambda v, start, length: grid_rows(axis_index - 1, {**bounds, v: (start, length)}),
                    budget,
                )
                rows = []
                for count, row in runs:
                    if len(row) == 1:
                        rows.append((count * row[0][0], row[0][1]))
                    else:
                        if len(rows) + count * len(row) > 4096:
                            raise UnresolvedRegion("unsupported_scheduling", "bounded launch schedule exceeded 4096 groups")
                        rows.extend(row * count)
                return [(sum(n for n, _ in group), body) for body, group in groupby(rows, key=lambda pair: pair[1])]

            schedules = grid_rows(len(axes) - 1, domains)
            repetitions = 1
    variants, groups = [], []
    for count, body in schedules:
        if body not in variants:
            variants.append(body)
        groups.append(dict(iterations=variants.index(body), count=count))
    return dict(
        variants=variants,
        cta_work=dict(precision="exact", groups=groups, repetitions=repetitions, grid_blocks=prod(n for _, n in domains.values())),
        partitions=4096 - budget[0],
    )


from tiletune_core.region_schedule import region_totals as region_totals


from tiletune_core.region_schedule import estimate_region_cycles as estimate_region_cycles
