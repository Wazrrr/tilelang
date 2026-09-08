"""Read-only IR queries shared by family recognition and resource models."""

from tvm import tirx as tir


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
