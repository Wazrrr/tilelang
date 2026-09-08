"""Derive work and compute participants from each captured IR operation.

These are operator primitives shared by GEMM and attention. A GEMM's work comes
from its native regions; reductions and scalar work come from their actual IR.
No kernel-family template or benchmark result supplies the workload.
"""

from math import prod
from tilelang import tvm
from tvm import tirx as tir


def operation_work(op):
    from .analysis import _int
    from .ir_utils import call_names

    work = dict(gemm_flops=0, elementwise_ops=0, exp_ops=0, reduction_ops=0, shared_bytes=0)

    def size(region):
        dims = [_int(r.extent) for r in region.region]
        return prod(dims) if all(v is not None for v in dims) else None

    meta = op.metadata
    if hasattr(meta, "cRegion"):
        elements = size(meta.cRegion)
        k = _int(meta.aRegion.region[0 if meta.transA else 1].extent)
        work["gemm_flops"] = 2 * elements * k if elements is not None and k is not None else None
        for region in (meta.aRegion, meta.bRegion):
            if region.buffer.scope().startswith("shared"):
                elements = size(region)
                dtype = tvm.DataType(region.buffer.dtype)
                if elements is None or work["shared_bytes"] is None:
                    work["shared_bytes"] = None
                else:
                    work["shared_bytes"] += (elements * dtype.bits * dtype.lanes + 7) // 8
    elif hasattr(meta, "dim") and hasattr(meta, "srcRegion"):
        src, dst = size(meta.srcRegion), size(meta.dstRegion)
        work["reduction_ops"] = src - dst if src is not None and dst is not None else None
    elif op.kind == "elementwise":
        # Parallel loops define one logical tile, not work per launch thread.
        dims = [_int(r.extent) for _, r, kind in op.loops if kind == str(tir.ForKind.PARALLEL)]
        elements = prod(dims) if all(v is not None for v in dims) else None
        names = call_names(op)
        exps = sum(name in ("tirx.exp", "tirx.exp2") for name in names)
        scalar_ops = []
        types = (tir.Add, tir.Sub, tir.Mul, tir.Div, tir.FloorDiv, tir.Max, tir.Min, tir.Cast, tir.Select)
        tir.stmt_functor.post_order_visit(op.metadata.value, lambda n: scalar_ops.append(n) if isinstance(n, types) else None)
        work["exp_ops"] = elements * exps if elements is not None else None
        work["elementwise_ops"] = elements * max(len(scalar_ops), 1) if elements is not None else None
        if any(name not in ("tirx.exp", "tirx.exp2", "tirx.if_then_else", "tirx.likely", "tl.infinity") for name in names):
            work["elementwise_ops"] = None
    elif op.kind in ("copy", "fill") and op.writes:
        # Register casts, initialization and epilogue stores are consumer work.
        # External read copies are charged to the producer separately.
        if not any(r.buffer.scope() == "global" for r in op.reads):
            dims = [_int(r.extent) for r in op.writes[0].ranges]
            work["elementwise_ops"] = prod(dims) if all(v is not None for v in dims) else None
    return work


def compute_participants(op, pressure, pass_configs):
    """Query instruction selection without lowering or modifying operator policy.

    A throughput ceiling per active warpgroup is distinct from the aggregate
    per-SM throughput ceiling. Automatic layout is not needed to count groups.
    """
    from .analysis import _int
    from tilelang.transform import PassContext
    from tvm.target import Target

    if not hasattr(op.metadata, "cRegion"):
        return None
    threads = [_int(v) for k, v in op.launch_threads.items() if k.startswith("threadIdx.")]
    if not threads or any(v is None or v <= 0 for v in threads) or not pressure.get("target_arch"):
        return {"precision": "unknown"}
    try:
        target = Target({"kind": "cuda", "arch": pressure["target_arch"]})
        registered = PassContext.list_configs()
        with PassContext(config={k: v for k, v in pass_configs.items() if k in registered}):
            instruction = op.metadata._select_gemm_instruction(prod(threads), target)
        return {
            "instruction": instruction,
            "consumer_threads": prod(threads),
            "a_dtype": str(op.metadata.a.dtype),
            "b_dtype": str(op.metadata.b.dtype),
            "accum_dtype": str(op.metadata.c.dtype),
            "warpgroups": prod(threads) // 128 if instruction == "cuda.wgmma" and prod(threads) % 128 == 0 else None,
            "precision": "predicted",
            "evidence": ["native GemmGetGemmInstructionKey on the original consumer thread domain; no lowering or layout inference"],
        }
    except Exception as error:
        return {"precision": "unknown", "reason": str(error)}


def consumer_threads(op):
    from .analysis import _int

    values = [_int(v) for k, v in op.launch_threads.items() if k.startswith("threadIdx.")]
    return prod(values) if values and all(v is not None and v > 0 for v in values) else None
