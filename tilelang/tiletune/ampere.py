"""Ampere software-pipeline plans and fragment ownership, before device compilation.

Compiler planning runs on a separate IRModule. The input PrimFunc and its
annotations are never changed. No candidate is compiled or timed by this module.
"""

from math import prod
from collections import OrderedDict
from copy import deepcopy

from tilelang import tvm, transform
from tvm import tirx as tir
from tvm.target import Target

from .src.collector import _Collector
from .src.structural_key import StructuralKey, context_key
from .src.ir_utils import _int, in_loop

_PREPARATION_CACHE = OrderedDict()


def is_ampere(target):
    from .targets import resolve_target

    return resolve_target(target).architecture == "ampere"


def prepare_analysis(func, col, target, pass_configs):
    """Reuse immutable facts only for an identical function and compiler context.

    The cache is process-local (the loaded native compiler identity is fixed),
    includes every IR expression, layout, shape, dtype, target and pass setting,
    and stores layouts by buffer position rather than retaining old buffer keys.
    These facts use no profile rates. No cache survives a compiler restart.
    """
    from .config import ANALYSIS_VERSION

    key = (ANALYSIS_VERSION, StructuralKey(func), str(Target(target)), context_key(pass_configs))
    cached = _PREPARATION_CACHE.get(key)
    if cached is not None:
        plan, plans, layouts, error = cached
        col.ampere_plan = deepcopy(plan)
        col.ampere_plans = {loop.loop_var: deepcopy(value) for loop, value in zip(col.pipeline_loops, plans) if value is not None}
        col.inferred_layouts = {col.buffers[i].data: layout for i, layout in layouts}
        if error:
            col.ampere_layout_unknown = error
        _PREPARATION_CACHE.move_to_end(key)
        return
    _prepare_analysis(func, col, target, pass_configs)
    _PREPARATION_CACHE[key] = (
        deepcopy(col.ampere_plan),
        [deepcopy(col.ampere_plans.get(loop.loop_var)) for loop in col.pipeline_loops],
        [(i, col.inferred_layouts[b.data]) for i, b in enumerate(col.buffers) if b.data in col.inferred_layouts],
        getattr(col, "ampere_layout_unknown", None),
    )
    if len(_PREPARATION_CACHE) > 64:
        _PREPARATION_CACHE.popitem(last=False)


def _prepare_analysis(func, col, target, pass_configs):
    """Keep backend observations separate from user-supplied layout annotations."""
    col.ampere_plan = {"status": "not_applicable", "events": [], "unknown": []}
    col.inferred_layouts = {}
    col.ampere_plans = {}
    registered = transform.PassContext.list_configs()
    with Target(target), transform.PassContext(config={k: v for k, v in pass_configs.items() if k in registered}):
        if col.pipeline_loops:
            try:
                col.ampere_plans = pipeline_plans(func, col, target)
                if len(col.ampere_plans) == 1:
                    col.ampere_plan = next(iter(col.ampere_plans.values()))
            except Exception as error:
                col.ampere_plan.update(status="unknown", unknown=[f"Ampere compiler pipeline plan: {error}"])
        # A producer's MMA layout alone does not determine the ownership of
        # subsequent broadcasts and reductions (notably online attention).
        # Resolve the complete graph before counting scalar or collective work.
        if any(op.kind == "reduce" for op in col.operations):
            try:
                from .ownership import verified_explicit_layouts

                explicit = verified_explicit_layouts(col, target, pass_configs)
                if explicit is not None:
                    col.inferred_layouts = explicit
                    return
                mod = tvm.IRModule({"main": func})
                mod = tir.transform.BindTarget(Target(target))(mod)
                for make_pass in (
                    transform.MaterializeKernelLaunch,
                    transform.AddWrapperForSingleBufStore,
                    transform.Simplify,
                    transform.LayoutReducer,
                    transform.LayoutInference,
                ):
                    mod = make_pass()(mod)

                def visit(node):
                    if isinstance(node, tir.SBlock):
                        for buffer, layout in node.annotations.get("layout_map", {}).items():
                            data = buffer.data if hasattr(buffer, "data") else buffer
                            if hasattr(layout, "replicate_size"):
                                col.inferred_layouts[data] = layout

                tir.stmt_functor.post_order_visit(mod["main"].body, visit)
                from .ownership import _verify_collective
                from .compute import consumer_threads

                for op in col.operations:
                    if op.kind == "reduce":
                        _verify_collective(op, col.inferred_layouts, consumer_threads(op), target, pass_configs)
            except Exception as error:
                col.ampere_layout_unknown = str(error)


def pipeline_plan(func, col, target):
    """Compatibility entry point for callers with a single pipeline."""
    plans = pipeline_plans(func, col, target)
    if len(plans) != 1:
        raise ValueError("expected a single software-pipeline plan")
    return next(iter(plans.values()))


def pipeline_plans(func, col, target):
    """Match each independently planned loop by its retained IR variable."""
    if col.unknown:
        raise ValueError("pipeline contains unresolved operations")
    mod = tir.transform.BindTarget(Target(target))(tvm.IRModule({"main": func}))
    mod = transform.IfStmtBinding()(mod)
    mod = transform.PipelinePlanning()(mod)
    loops = []
    tir.stmt_functor.post_order_visit(
        mod["main"].body,
        lambda node: loops.append(node) if isinstance(node, tir.For) and "software_pipeline_stage" in node.annotations else None,
    )
    plans = {}
    for original in col.pipeline_loops:
        matches = [loop for loop in loops if loop.loop_var.same_as(original.loop_var)]
        if len(matches) != 1:
            raise ValueError("compiler did not retain a unique pipeline loop identity")
        plans[original.loop_var] = _loop_plan(func, col, original, matches[0])
    return plans


def _loop_plan(func, col, original_loop, loop):
    depth = _int(original_loop.annotations.get("num_stages"))
    if depth is None or not 1 <= depth <= 32:
        raise ValueError("requires a constant stage count between 1 and 32")
    stages = [int(x) for x in loop.annotations["software_pipeline_stage"]]
    orders = [int(x) for x in loop.annotations["software_pipeline_order"]]
    groups = [int(x) for x in loop.annotations.get("software_pipeline_async_producer_groups", [-1] * len(stages))]
    body = loop.body
    if isinstance(body, tir.SBlockRealize):
        body = body.block.body
    statements = list(body.seq) if isinstance(body, tir.SeqStmt) else [body]
    if len(statements) != len(stages):
        raise ValueError("compiler stage annotations do not match the statement sequence")
    inside = [op for op in col.operations if in_loop(op, original_loop)]
    position, events = 0, []
    for statement, stage, order, group in zip(statements, stages, orders, groups):
        captured = _Collector(func.with_body(statement)).operations
        matched = inside[position : position + len(captured)]
        if len(matched) != len(captured) or not captured:
            raise ValueError("unresolved mapping from planned statements to original operations")
        for old, new in zip(matched, captured):
            if (
                old.kind != new.kind
                or len(old.reads) != len(new.reads)
                or any(not a.buffer.data.same_as(b.buffer.data) for a, b in zip(old.reads, new.reads))
                or len(old.writes) != len(new.writes)
                or any(not a.buffer.data.same_as(b.buffer.data) for a, b in zip(old.writes, new.writes))
            ):
                raise ValueError("compiler planning changed operation identity")
        events.append(dict(operations=[op.index for op in matched], stage=stage, order=order, async_group=group))
        position += len(captured)
    if position != len(inside):
        raise ValueError("compiler plan does not cover every original operation")
    return dict(
        status="predicted",
        events=sorted(events, key=lambda event: event["order"]),
        max_stage=max(stages),
        buffer_depth=depth,
        unknown=[],
        assumptions=[
            "native PipelinePlanning stage/order/async-group annotations on an isolated IRModule",
            "ordinary consumer warps issue copies and compute; no added producer warpgroup",
            "copy vectorization and instruction issue remain effective-service estimates",
        ],
    )


def external_work(op):
    """Bytes accessed by one tile operation, preserving the recurrence coordinate."""
    from .src.ir_utils import _domains, rectangular_scalar_access
    from .src.regions import _bound, _clip_global

    domains = _domains(tuple(axis for axis in op.loops if axis[2] == "1"))
    reads, writes = [], []
    for regions, output in ((op.reads, reads), (op.writes, writes)):
        seen = set()
        for region in regions:
            if region.buffer.scope() != "global":
                continue
            if op.kind == "elementwise" and not rectangular_scalar_access(region, op.loops):
                return {"read_bytes": None, "write_bytes": None, "read_groups": None}
            bounded = _clip_global(_bound(region, domains), op.loops)
            dims = [_int(r.extent) for r in bounded.ranges]
            key = (region.buffer, tuple((StructuralKey(r.min), StructuralKey(r.extent)) for r in bounded.ranges))
            if key in seen:
                continue
            seen.add(key)
            if any(n is None or n < 0 for n in dims):
                return {"read_bytes": None, "write_bytes": None, "read_groups": None}
            dtype = tvm.DataType(region.buffer.dtype)
            output.append((prod(dims) * dtype.bits * dtype.lanes + 7) // 8)
    return dict(read_bytes=sum(reads), write_bytes=sum(writes), read_groups=len(reads))


def operand_registers(col, target):
    """MMA lowering allocates temporary operand fragments for shared operands."""
    from .compute import consumer_threads
    from tilelang.cuda.op.gemm.gemm_mma import GemmMMA

    result = {}
    for op in col.operations:
        if not hasattr(op.metadata, "cRegion"):
            continue
        threads = consumer_threads(op)
        try:
            emitter = GemmMMA(op.metadata)._make_mma_emitter(Target(target), threads)
            registers = 0
            for buffer, slots in (
                (op.metadata.a, emitter.warp_rows * emitter.local_size_a),
                (op.metadata.b, emitter.warp_cols * emitter.local_size_b),
            ):
                if buffer.scope().startswith("shared"):
                    dtype = tvm.DataType(buffer.dtype)
                    registers += (int(slots) * dtype.bits * dtype.lanes + 31) // 32
            result[op.index] = dict(registers_per_thread=registers, registers_per_block=registers * threads)
        except Exception as error:
            result[op.index] = dict(unknown=str(error))
    return result


from tiletune_core.ampere import schedule_cycles as schedule_cycles
