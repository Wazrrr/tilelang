"""Ampere software-pipeline plans and fragment ownership, before device compilation.

Compiler planning runs on a separate IRModule. The input PrimFunc and its
annotations are never changed. No candidate is compiled or timed by this module.
"""

from math import prod

from tilelang import tvm, transform
from tvm import tirx as tir
from tvm.target import Target

from .src.collector import _Collector
from .src.ir_utils import _int, in_loop


def is_ampere(target):
    from .targets import resolve_target

    return resolve_target(target).architecture == "ampere"


def prepare_analysis(func, col, target, pass_configs):
    """Keep backend observations separate from user-supplied layout annotations."""
    col.ampere_plan = {"status": "not_applicable", "events": [], "unknown": []}
    col.inferred_layouts = {}
    registered = transform.PassContext.list_configs()
    with Target(target), transform.PassContext(config={k: v for k, v in pass_configs.items() if k in registered}):
        if col.pipeline_loops:
            try:
                col.ampere_plan = pipeline_plan(func, col, target)
            except Exception as error:
                col.ampere_plan.update(status="unknown", unknown=[f"Ampere compiler pipeline plan: {error}"])
        # MMA layout helpers already cover dense producers. Generic reductions
        # need the compiler's graph-wide layout constraints, including broadcasts
        # and loop-carried state. These serial/single-pass graphs are inexpensive
        # to infer and require no software-pipeline rewrite.
        if any(op.kind == "reduce" for op in col.operations) and not any(hasattr(op.metadata, "cRegion") for op in col.operations):
            try:
                if col.pipeline_loops:
                    raise ValueError("generic reduction ownership in a software pipeline is not resolved")
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
            except Exception as error:
                col.ampere_layout_unknown = str(error)


def pipeline_plan(func, col, target):
    if len(col.pipeline_loops) != 1 or col.unknown or any(op.predicates for op in col.operations):
        raise ValueError("requires one pipeline with resolved, unconditional operations")
    original_loop = col.pipeline_loops[0]
    depth = _int(original_loop.annotations.get("num_stages"))
    if depth is None or not 1 <= depth <= 32:
        raise ValueError("requires a constant stage count between 1 and 32")
    mod = tir.transform.BindTarget(Target(target))(tvm.IRModule({"main": func}))
    mod = transform.IfStmtBinding()(mod)
    mod = transform.PipelinePlanning()(mod)
    loops = []
    tir.stmt_functor.post_order_visit(
        mod["main"].body,
        lambda node: loops.append(node) if isinstance(node, tir.For) and "software_pipeline_stage" in node.annotations else None,
    )
    if len(loops) != 1:
        raise ValueError("compiler did not produce one software-pipeline plan")
    loop = loops[0]
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
    from .src.ir_utils import _domains
    from .src.regions import _bound, _clip_global

    domains = _domains(tuple(axis for axis in op.loops if axis[2] == "1"))
    reads, writes = [], []
    for regions, output in ((op.reads, reads), (op.writes, writes)):
        seen = set()
        for region in regions:
            if region.buffer.scope() != "global":
                continue
            bounded = _clip_global(_bound(region, domains), op.loops)
            dims = [_int(r.extent) for r in bounded.ranges]
            key = (hash(region.buffer), str(bounded.to_dict()["ranges"]))
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


def schedule_cycles(plan, copies, phase_cycles, iterations, profile, concurrent_ctas):
    """Replay a periodic compiler plan with max-plus transitions, not loop unrolling.

    Copy issue and consumers share one ordered instruction stream. Memory byte
    service proceeds asynchronously. Ready-time history follows the compiler's
    iteration offsets; group waits precede first use. Boundary segments cover
    prologue, steady state and drain, including loops shorter than the pipeline.
    """
    from .schedule import NEG, _maximum, _delay, repeat_transition

    if iterations == 0:
        return 0.0
    bandwidth = profile["global_bytes_per_cycle"] / concurrent_ctas
    issue_rate = profile.get("async_copy_issue_bytes_per_cycle")
    latency = profile.get("async_copy_latency_cycles")
    if not issue_rate or latency is None:
        return None
    events = plan["events"]
    by_op = {copy["operation"]: copy for copy in copies}
    copy_events = {op: event for event in events for op in event["operations"] if op in by_op}
    if len(copy_events) != len(copies):
        return None
    distance = max(1, plan["max_stage"])
    copy_ids = list(by_op)
    offsets = {op: 2 + i * distance for i, op in enumerate(copy_ids)}
    size = 2 + len(copies) * distance
    basis = [[0.0 if i == j else NEG for j in range(size)] for i in range(size)]
    state = [0.0] * size
    boundaries = sorted(
        {0, iterations + plan["max_stage"], *(event["stage"] for event in events), *(iterations + event["stage"] for event in events)}
    )
    for start, end in zip(boundaries, boundaries[1:]):
        warp, service = basis[:2]
        produced, waited = {}, set()
        for event in events:
            if not 0 <= start - event["stage"] < iterations:
                continue
            for op in event["operations"]:
                if op in by_op:
                    copy = by_op[op]
                    if event["async_group"] < 0:
                        warp = _delay(warp, copy["bytes"] / bandwidth + profile["copy_latency_cycles"])
                        produced[op] = warp
                    else:
                        warp = _delay(warp, copy["bytes"] * concurrent_ctas / issue_rate)
                        service = _delay(_maximum(warp, service), copy["bytes"] / bandwidth)
                        produced[op] = _maximum(service, _delay(warp, latency))
                    continue
                required = [cid for cid, copy in by_op.items() if copy["first_consumer"] == op]
                groups = {copy_events[cid]["async_group"] for cid in required}
                for group in groups:
                    key = (group, event["stage"])
                    if key in waited:
                        continue
                    members = [cid for cid in copy_ids if copy_events[cid]["async_group"] == group] if group >= 0 else required
                    ready = []
                    for cid in members:
                        age = event["stage"] - copy_events[cid]["stage"]
                        if age < 0 or age > distance:
                            return None
                        if age == 0:
                            if cid not in produced:
                                return None
                            ready.append(produced[cid])
                        else:
                            ready.append(basis[offsets[cid] + age - 1])
                    warp = _delay(_maximum(warp, *ready), profile["barrier_cycles"])
                    waited.add(key)
                warp = _delay(warp, phase_cycles[op])
        updated = [warp, service]
        for cid in copy_ids:
            updated.append(produced.get(cid, [NEG] * size))
            updated.extend(basis[offsets[cid] + age] for age in range(distance - 1))
        matrix = tuple(tuple(row) for row in updated)
        state = repeat_transition(matrix, end - start, state=state)
    return state[0]
