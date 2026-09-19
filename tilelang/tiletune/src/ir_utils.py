"""Read-only arithmetic, control-flow and operation queries."""

from math import prod
from tilelang import tvm
from tvm import tirx as tir
from tvm.arith import Analyzer


def _int(expr):
    try:
        return int(Analyzer().simplify(expr))
    except (TypeError, ValueError):
        return None


def _exclusive(a, b):
    return any(i == j and x != y for i, x in a.branches for j, y in b.branches)


def _domains(loops):
    return {v: tvm.arith.IntervalSet(r.min, r.min + r.extent - 1) for v, r, _ in loops}


def resolve_pass_configs(func, pass_configs):
    """Resolve compiler settings once at the analysis boundary."""
    from tilelang.transform import PassContext

    effective = dict(PassContext.current().config)
    effective.update(dict((func.attrs or {}).get("tilelang_pass_configs", {})))
    effective.update(pass_configs or {})
    return effective


def in_loop(op, loop):
    return loop is not None and any(v.same_as(loop.loop_var) for v, _, _ in op.loops)


def rectangular_scalar_access(region, loops):
    """Prove a scalar tile describes a dense rectangular set of addresses.

    Bounding boxes alone overcount strided and diagonal accesses. Such mappings
    need a separate cardinality model and must not become exact byte counts.
    """
    axes = {v: domain for v, domain, kind in loops if kind == "1"}
    analyzer = Analyzer()
    for var, domain in axes.items():
        analyzer.bind(var, domain)
    assigned = set()
    for interval in region.ranges:
        used = set()
        tir.stmt_functor.post_order_visit(interval.min, lambda n, used=used: used.add(n) if isinstance(n, tir.Var) and n in axes else None)
        if not used:
            continue
        if len(used) != 1 or used & assigned:
            return False
        var = next(iter(used))
        assigned.add(var)
        base = tir.stmt_functor.substitute(interval.min, {var: tir.IntImm(var.dtype, 0)})
        one = tir.stmt_functor.substitute(interval.min, {var: tir.IntImm(var.dtype, 1)})
        stride = _int(analyzer.simplify(one - base))
        if stride not in (-1, 1) or not analyzer.can_prove(interval.min == base + stride * var):
            return False
    return True


def dense_gemms(col):
    return [op for op in col.operations if hasattr(op.metadata, "cRegion")]


def main_loops(col, operations):
    loops = [loop for loop in col.pipeline_loops if all(in_loop(op, loop) for op in operations)]
    # T.Pipelined(num_stages=0) elaborates to an ordinary serial loop. Recognize
    # its actual copy/compute loop without requiring a disappeared annotation.
    return loops or [loop for loop in col.serial_loops if all(in_loop(op, loop) for op in operations)]


def call_names(op):
    names = []
    if op.kind == "elementwise":
        tir.stmt_functor.post_order_visit(
            op.metadata.value, lambda n: names.append(str(n.op.name)) if isinstance(n, tir.Call) and hasattr(n.op, "name") else None
        )
    return names


def loop_visits(loops):
    expression = prod(r.extent for _, r, kind in loops if kind != "4")
    value = _int(expression)
    if value is not None:
        precision = "conservative" if any(kind == "while_bound" for _, _, kind in loops) else "exact"
        return {"min": value, "max": value, "expression": str(expression), "precision": precision}
    bounds = Analyzer().int_set(expression, _domains(loops))
    lower, upper = _int(bounds.min_value), _int(bounds.max_value)
    return {
        "min": lower,
        "max": upper,
        "expression": str(expression),
        "precision": "conservative" if lower is not None and upper is not None else "unknown",
    }
