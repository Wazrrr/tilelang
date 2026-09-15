"""Verified explicit fragment ownership without redundant graph inference."""

from collections import OrderedDict

from tilelang import tvm, transform
import tilelang.language as T
from tvm import tirx as tir
from tvm.arith import Analyzer
from tvm.ir import Range
from tvm.target import Target

from .src.ir_utils import _int
from .src.structural_key import StructuralKey, context_key

_COLLECTIVES = OrderedDict()


def _mapping(layout, indices):
    variables = list(layout.get_forward_vars())[-len(indices) :]
    thread = tir.stmt_functor.substitute(layout.thread, dict(zip(variables, indices)))
    used = set()
    tir.stmt_functor.post_order_visit(layout.thread, lambda n: used.add(n) if isinstance(n, tir.Var) else None)
    replica = [v for v in used if v not in variables]
    if len(replica) > 1:
        raise ValueError("unresolved explicit replication")
    return thread, replica[0] if replica else None


def _contains_owner(layout, indices, owner, analyzer):
    thread, replica = _mapping(layout, indices)
    if replica is None:
        return analyzer.can_prove(thread == owner)
    base = tir.stmt_functor.substitute(thread, {replica: tir.IntImm(replica.dtype, 0)})
    next_thread = tir.stmt_functor.substitute(thread, {replica: tir.IntImm(replica.dtype, 1)})
    stride = _int(analyzer.simplify(next_thread - base))
    if not stride or stride < 0 or not analyzer.can_prove(thread == base + replica * stride):
        return False
    delta = owner - base
    return all(analyzer.can_prove(p) for p in (delta >= 0, delta < layout.replicate_size * stride, tir.floormod(delta, stride) == 0))


def _verify_collective(op, layouts, threads, target, pass_configs):
    meta = op.metadata
    source, destination = layouts[meta.src.data], layouts[meta.dst.data]
    shape, output = tuple(_int(n) for n in meta.src.shape), tuple(_int(n) for n in meta.dst.shape)
    axis, batch, kind = _int(meta.dim), _int(meta.batch), int(meta.type.type)
    key = (
        StructuralKey(source),
        StructuralKey(destination),
        shape,
        output,
        str(meta.src.dtype),
        str(meta.dst.dtype),
        axis,
        batch,
        kind,
        threads,
        str(Target(target)),
        context_key(pass_configs),
    )
    if key in _COLLECTIVES:
        return
    if kind not in (0, 2) or batch != 1:
        raise ValueError("unmodeled explicit collective")
    src_dtype, dst_dtype = str(meta.src.dtype), str(meta.dst.dtype)

    @T.prim_func
    def primitive():
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment(shape, src_dtype)
            dst = T.alloc_fragment(output, dst_dtype)
            T.annotate_layout({src: source, dst: destination})
            if kind == 0:
                T.reduce_sum(src, dst, dim=axis)
            else:
                T.reduce_max(src, dst, dim=axis)

    mod = tir.transform.BindTarget(Target(target))(tvm.IRModule({"main": primitive}))
    for make_pass in (
        transform.MaterializeKernelLaunch,
        transform.AddWrapperForSingleBufStore,
        transform.Simplify,
        transform.LayoutReducer,
        transform.LayoutInference,
        transform.LowerTileOp,
    ):
        mod = make_pass()(mod)
    calls = []

    def collect(node):
        if isinstance(node, tir.Call) and getattr(node.op, "name", "") == "tirx.call_extern":
            name = str(node.args[0])
            if "AllReduce" in name:
                if "::run_batch" in name or "::run" not in name:
                    raise ValueError("compiler selected an unsupported collective")
                calls.append(name)

    tir.stmt_functor.post_order_visit(mod["main"].body, collect)
    _COLLECTIVES[key] = tuple(calls)
    if len(_COLLECTIVES) > 256:
        _COLLECTIVES.popitem(last=False)


def verified_explicit_layouts(col, target, pass_configs):
    """Prove each local access stays on its owner; ask the native reduction
    inferencer to verify each distinct collective. General graphs fall back to
    full LayoutInference. No BufferRef is stored in the collective cache.
    """
    from .compute import consumer_threads

    fragments = [b for b in col.buffers if b.scope() == "local.fragment"]
    if not fragments or col.unknown:
        return None
    layouts = {b.data: col.layouts[b.data] for b in fragments if b.data in col.layouts}
    try:
        # Dense MMA producers fix their accumulator mapping through the same
        # helper used by lowering. Shared intermediates impose no cross-thread
        # fragment ownership constraint on later scalar consumers.
        from tilelang.cuda.op.gemm.gemm_mma import GemmMMA

        for op in col.operations:
            if not hasattr(op.metadata, "cRegion"):
                continue
            threads = consumer_threads(op)
            if op.metadata._select_gemm_instruction(threads, Target(target)) != "cuda.mma":
                return None
            produced = GemmMMA(op.metadata).infer_layout(Target(target), threads)
            for buffer, layout in produced.items():
                if buffer.scope() != "local.fragment":
                    continue
                if buffer.data in layouts and not tvm.ir.structural_equal(layouts[buffer.data], layout):
                    return None
                layouts[buffer.data] = layout
        if any(b.data not in layouts for b in fragments):
            return None
        for layout in layouts.values():
            layout.inverse()
        for op in col.operations:
            if hasattr(op.metadata, "cRegion"):
                continue
            if op.kind == "reduce":
                _verify_collective(op, layouts, consumer_threads(op), target, pass_configs)
            elif op.kind == "elementwise":
                if op.metadata.buffer.data not in layouts:
                    return None
                layout = layouts[op.metadata.buffer.data]
                owner, replica = _mapping(layout, list(op.metadata.indices))
                analyzer = Analyzer()
                for var, domain, _ in op.loops:
                    analyzer.bind(var, domain)
                if replica is not None:
                    analyzer.bind(replica, Range.from_min_extent(0, layout.replicate_size))
                loads = []
                tir.stmt_functor.post_order_visit(
                    op.metadata.value, lambda n, loads=loads: loads.append(n) if isinstance(n, tir.BufferLoad) else None
                )
                for load in loads:
                    if load.buffer.scope().startswith("shared"):
                        continue
                    if load.buffer.data not in layouts or not _contains_owner(
                        layouts[load.buffer.data], list(load.indices), owner, analyzer
                    ):
                        return None
            elif op.kind in ("copy", "async_copy", "fill"):
                local = [r for r in op.reads + op.writes if r.buffer.scope() != "global" and not r.buffer.scope().startswith("shared")]
                if any(r.buffer.data not in layouts for r in local):
                    return None
                if op.reads and op.writes and all(r.buffer.scope() == "local.fragment" for r in op.reads + op.writes):
                    for region in op.reads + op.writes:
                        if not tvm.ir.structural_equal(layouts[region.buffer.data], layouts[op.writes[0].buffer.data]) or any(
                            _int(r.min) != 0 or _int(r.extent) != _int(n) for r, n in zip(region.ranges, region.buffer.shape)
                        ):
                            return None
            else:
                return None
    except Exception:
        return None
    return layouts
