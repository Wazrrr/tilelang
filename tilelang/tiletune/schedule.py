"""Buffer readiness/reuse recurrence and launch-wide CTA scheduling."""

from math import inf, prod
from itertools import groupby
from tilelang import tvm
from tvm import tirx as tir
from tvm.arith import Analyzer
from .src.ir_utils import _int, in_loop

NEG = -inf


from tiletune_core.schedule import _maximum as _maximum


from tiletune_core.schedule import _delay as _delay


from tiletune_core.schedule import _square as _square


from tiletune_core.schedule import repeat_transition as repeat_transition


from tiletune_core.schedule import buffer_transition as buffer_transition


def collect_producer_buffers(col, loop):
    """Read actual copy regions and consumers; do not infer operand roles by name."""

    inside = [op for op in col.operations if in_loop(op, loop)]
    producers = [
        op
        for op in inside
        if op.kind in ("copy", "async_copy")
        and any(r.buffer.scope() == "global" for r in op.reads)
        and any(r.buffer.scope().startswith("shared") for r in op.writes)
    ]
    copies, targets = [], set()
    for op in producers:
        if len(op.writes) != 1 or op.unknown or op.predicates:
            raise ValueError("unresolved producer destination or conditional copy")
        region = op.writes[0]
        buffer = region.buffer
        if buffer in targets:
            raise ValueError("multiple producer writes to one buffer require a region-aware reuse schedule")
        targets.add(buffer)
        uses = [other.index for other in inside if any(r.buffer.same_as(buffer) for r in other.reads)]
        if (
            not uses
            or min(uses) <= op.index
            or any(other is not op and any(r.buffer.same_as(buffer) for r in other.writes) for other in inside)
        ):
            raise ValueError("unresolved producer/consumer reuse or overwrite")
        dims = [_int(r.extent) for r in region.ranges]
        if any(x is None or x <= 0 for x in dims):
            raise ValueError("symbolic producer tile size")
        dtype = tvm.DataType(buffer.dtype)
        sources = []
        for read in op.reads:
            if read.buffer.scope() == "global" and not any(read.buffer.same_as(item) for item in sources):
                sources.append(read.buffer)
        source = sources[0] if len(sources) == 1 else None
        source_shape = [_int(value) for value in source.shape] if source is not None else []
        source_dtype = tvm.DataType(source.dtype) if source is not None else None
        source_bytes = (
            (prod(source_shape) * source_dtype.bits * source_dtype.lanes + 7) // 8
            if source is not None and all(value is not None and value > 0 for value in source_shape)
            else None
        )
        copies.append(
            dict(
                operation=op.index,
                buffer=buffer.name,
                buffer_id=str(hash(buffer)),
                region=region.to_dict(),
                dtype=str(buffer.dtype),
                bytes=(prod(dims) * dtype.bits * dtype.lanes + 7) // 8,
                source_buffer=source.name if source is not None else None,
                source_buffer_id=str(hash(source)) if source is not None else None,
                source_bytes=source_bytes,
                first_consumer=min(uses),
                last_consumer=max(uses),
            )
        )
    return copies


def collect_cta_work(col, loop, max_axis_points=4096):
    """Compress one varying block axis into runs; never expand the inner loop."""

    domains = col.block_domains
    result = {"precision": "unknown", "groups": [], "repetitions": None, "grid_blocks": None, "unknown": []}
    if (loop is None and (col.pipeline_loops or col.serial_loops)) or not domains:
        result["unknown"].append("no unique launch domain and pipeline loop")
        return result
    axes = sorted(domains)
    sizes = [_int(domains[axis][1].extent) for axis in axes]
    if any(n is None or n <= 0 for n in sizes):
        result["unknown"].append("symbolic CTA launch domain")
        return result
    result["grid_blocks"] = prod(sizes)
    extent = loop.extent if loop is not None else tir.IntImm("int32", 0)
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


from tiletune_core.schedule import estimate_grid_cycles as estimate_grid_cycles
