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


def dense_gemms(col):
    return [op for op in col.operations if hasattr(op.metadata, "cRegion") and not bool(getattr(op.metadata, "isTcgen05", False))]


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
        return {"min": value, "max": value, "expression": str(expression), "precision": "exact"}
    bounds = Analyzer().int_set(expression, _domains(loops))
    lower, upper = _int(bounds.min_value), _int(bounds.max_value)
    return {
        "min": lower,
        "max": upper,
        "expression": str(expression),
        "precision": "conservative" if lower is not None and upper is not None else "unknown",
    }
