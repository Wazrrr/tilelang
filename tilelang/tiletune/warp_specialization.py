"""Pre-lowering Hopper scheduling policy for simple tile pipelines.

The native copy query is the same read-only classifier used by the WS pass.
This model deliberately requires a straight-line, pure-TMA producer pipeline;
manual WS, mixed producers and unresolved control flow need further modeling.
Register requests are policy reservations, not compiler register counts.
"""

import tvm_ffi
from tvm import tirx as tir
from tvm.target import Target
from tilelang.transform import PassContext


def predict_warp_specialization(func, col, pressure, pass_configs, *, specialization):
    from .analysis import _int

    effective = pass_configs
    result = {
        "status": "not_applicable",
        "applies": False,
        "producer_threads": None,
        "consumer_threads": None,
        "launch_threads": None,
        "producer_register_request": None,
        "consumer_register_request": None,
        "register_reservation_per_block": None,
        "compiler_registers_per_thread": None,
        "evidence": [],
        "effective_pass_configs": {str(k): str(v) for k, v in effective.items()},
    }

    def unknown(reason):
        result.update(status="unknown", applies=None)
        result["evidence"].append(reason)
        return result

    if pressure["target_arch"] not in ("sm_90", "sm_90a"):
        result["evidence"].append("Hopper policy model does not apply to this target")
        return result
    if bool(effective.get("tl.disable_warp_specialized", False)):
        result.update(status="disabled")
        result["evidence"].append("effective tl.disable_warp_specialized=true")
        return result

    nodes = []
    tir.stmt_functor.post_order_visit(func.body, nodes.append)
    if any(isinstance(n, tir.AttrStmt) and ("warp_special" in n.attr_key.lower() or "WarpSpecial" in n.attr_key) for n in nodes):
        return unknown("manual warp specialization requires explicit partition analysis")
    loops = [n for n in nodes if isinstance(n, tir.For) and "num_stages" in n.annotations]
    if any(_int(n.annotations["num_stages"]) is None for n in loops):
        return unknown("unresolved pipeline stages")
    loops = [n for n in loops if _int(n.annotations["num_stages"]) > 0]
    if not loops:
        result["evidence"].append("no positive-stage pipeline: automatic WS pass does not apply")
        return result
    if len(loops) != 1 or col.unknown or any(op.unknown or op.predicates for op in col.operations):
        return unknown("multiple pipelines, opaque operations, aliases or predicates")
    if col.layouts:
        return unknown("explicit layouts require checking compatibility with multi-version buffering")
    loop = loops[0]
    body = loop.body
    if isinstance(body, tir.SBlockRealize):
        body = body.block.body
    statements = list(body.seq) if isinstance(body, tir.SeqStmt) else [body]
    policy = specialization.warp_specialization_policy()
    if policy.require_tile_calls and not all(isinstance(s, tir.Evaluate) and isinstance(s.value, tir.Call) for s in statements):
        return unknown("pipeline body is not a straight-line sequence of tile operations")
    pipeline = [op for op in col.operations if any(v.same_as(loop.loop_var) for v, _, _ in op.loops)]
    if (policy.require_tile_calls and len(pipeline) != len(statements)) or not any(hasattr(op.metadata, "cRegion") for op in pipeline):
        return unknown("unresolved pipeline structure or consumer tile")
    domains = {tuple(sorted((k, str(v)) for k, v in op.launch_threads.items())) for op in col.operations}
    threads = _int(col.threads.get("threadIdx.x"))
    if (
        len(domains) != 1
        or threads not in (128, 256, 384)
        or any(_int(v) != 1 for k, v in col.threads.items() if k in ("threadIdx.y", "threadIdx.z"))
    ):
        return unknown("unsupported or unresolved producer/consumer thread partition")
    copies = [
        op
        for op in pipeline
        if op.kind in ("copy", "async_copy")
        and hasattr(op.metadata, "src")
        and op.metadata.src.scope() == "global"
        and op.metadata.dst.scope().startswith("shared")
    ]
    consumers = [op for op in pipeline if not any(op is copy for copy in copies)]
    allowed = all(policy.accepts_consumer(op, loop) for op in consumers)
    if not copies or not allowed:
        return unknown("pipeline includes consumer operations outside the modeled dense GEMM path")
    globals_ = list(func.buffer_map.values())
    if any(not any(op.metadata.src.same_as(b) for b in globals_) or not op.metadata.dst.scope().startswith("shared") for op in copies):
        return unknown("producer copies require direct parameter-buffer to shared-buffer access")
    classify = tvm_ffi.get_global_func("tl.tiletune.ClassifyProducerCopy")
    registered = PassContext.list_configs()
    with PassContext(config={k: v for k, v in effective.items() if k in registered}):
        target = Target({"kind": "cuda", "arch": pressure["target_arch"]})
        for op in copies:
            choice = classify(op.metadata, target)
            result["evidence"].append(f"operation {op.index}: native producer classifier={choice['kind']}")
            if not bool(choice["supported"]) or str(choice["kind"]) != "tma":
                return unknown("non-TMA or mixed producer path; " + str(choice["reason"]))

    # ProducerConsumerWSRewriter::BuildWSBlock uses 128 threads for pure TMA.
    # AnnotateWarpGroupRegAlloc uses 24/240 for 1P+1C and 1P+2C, 24/160
    # for 1P+3C. These requests do not scale with the pipeline ring size.
    producer, consumer = 24, 160 if threads == 384 else 240
    result.update(
        status="predicted",
        applies=True,
        producer_threads=128,
        consumer_threads=threads,
        launch_threads=128 + threads,
        producer_register_request=producer,
        consumer_register_request=consumer,
        register_reservation_per_block=128 * producer + threads * consumer,
    )
    result["evidence"] += [
        "pure-TMA pipeline with automatic layouts and recognized " + policy.consumer_description,
        "ProducerConsumerWSRewriter: one added producer warpgroup, original threads remain consumers",
        "AnnotateWarpGroupRegAlloc: default partition-dependent producer/consumer register requests",
        "reservation = producer_threads * producer_request + consumer_threads * consumer_request",
        "compiler initial allocation, allocation granularity and pre-partition scratch remain unmodeled",
    ]
    return result
