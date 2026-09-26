"""Lean logical compute-work capture without layout or schedule inference."""

from math import prod

from tvm import tirx as tir

from tiletune_core.work_max import WORK_KINDS
from .compute import compute_participants, operation_work
from .src.ir_utils import _int, loop_visits


_NON_ARITHMETIC = {
    "barrier",
    "tma_copy",
    "tirx.assume",
    "tirx.ptx_arrive_barrier",
    "tl.tcgen05_mma_arrive",
    "tl.fence_proxy_async",
    "tl.ptx_arrive_cluster_barrier",
    "tl.ptx_tcgen05_cp_warpx4",
    "tl.ptx_tcgen05_sf_warp_transpose",
}


def _scalar_work(expression):
    """Count the value tree, treating buffer loads and their indices as leaves."""
    work = dict(elementwise_ops=0, exp_ops=0, rsqrt_ops=0)
    unknown = []
    binary = (
        tir.Add,
        tir.Sub,
        tir.Mul,
        tir.Div,
        tir.FloorDiv,
        tir.FloorMod,
        tir.Min,
        tir.Max,
        tir.LT,
        tir.LE,
        tir.GT,
        tir.GE,
        tir.EQ,
        tir.NE,
        tir.And,
        tir.Or,
    )

    def visit(node):
        if isinstance(node, (tir.BufferLoad, tir.IntImm, tir.FloatImm, tir.StringImm, tir.Var)):
            return
        if isinstance(node, binary):
            work["elementwise_ops"] += 1
            visit(node.a)
            visit(node.b)
        elif isinstance(node, (tir.Cast, tir.Not)):
            work["elementwise_ops"] += 1
            visit(node.value if isinstance(node, tir.Cast) else node.a)
        elif isinstance(node, tir.Select):
            work["elementwise_ops"] += 1
            visit(node.condition)
            visit(node.true_value)
            visit(node.false_value)
        elif isinstance(node, tir.Call):
            name = node.op.name if hasattr(node.op, "name") else str(node.op)
            if name in ("tirx.exp", "tirx.exp2"):
                work["exp_ops"] += 1
            elif name == "tirx.rsqrt":
                work["rsqrt_ops"] += 1
            elif name in (
                "tirx.if_then_else",
                "tirx.bitwise_and",
                "tirx.bitwise_or",
                "tirx.bitwise_xor",
                "tirx.shift_left",
                "tirx.shift_right",
            ):
                work["elementwise_ops"] += 1
            elif name not in ("tirx.likely", "tl.infinity"):
                unknown.append(f"unsupported scalar call {name}")
            for argument in node.args:
                visit(argument)
        else:
            unknown.append(f"unsupported scalar expression {type(node).__name__}")

    visit(expression)
    work["elementwise_ops"] = max(1, work["elementwise_ops"])
    return work, unknown


def analyze_compute_work(col, pressure, pass_configs):
    """Capture padded logical work; count parallel and serial visits only once."""
    totals = dict.fromkeys(WORK_KINDS, 0)
    unknown = [reason for reason in col.unknown if not reason.startswith("operation ") and reason != "unresolved data-dependent binding"]
    operations, signatures, reduction_dtypes, omitted = [], [], set(), set()
    for operation in col.operations:
        if operation.kind in _NON_ARITHMETIC:
            omitted.add(operation.kind)
            continue
        visits = loop_visits(operation.loops)["max"]
        work = {}
        metadata = operation.metadata
        if hasattr(metadata, "cRegion"):
            participants = compute_participants(operation, pressure, pass_configs)
            instruction = participants.get("instruction") if participants else None
            if instruction is None:
                unknown.append(f"operation {operation.index}: unresolved matrix instruction")
            else:
                signature = {key: participants[key] for key in ("instruction", "a_dtype", "b_dtype", "accum_dtype")}
                if signature not in signatures:
                    signatures.append(signature)
            kind = "tcgen05_gemm_flops" if instruction == "cuda.tcgen05" else "gemm_flops"
            work[kind] = operation_work(operation)["gemm_flops"]
        elif operation.kind == "elementwise":
            work, reasons = _scalar_work(metadata.value)
            unknown.extend(f"operation {operation.index}: {reason}" for reason in reasons)
        elif operation.kind == "reduce" and hasattr(metadata, "srcRegion"):
            reduction_dtypes.add(str(metadata.src.dtype))
            reduction_kind = int(metadata.type.type)
            if reduction_kind not in (0, 2):
                unknown.append(f"operation {operation.index}: unsupported reduction kind {reduction_kind}")
            else:
                amount = operation_work(operation)["reduction_ops"]
                if not bool(metadata.clear):
                    extents = [_int(axis.extent) for axis in metadata.dstRegion.region]
                    amount = amount + prod(extents) if amount is not None and all(value is not None for value in extents) else None
                work["reduction_ops" if reduction_kind == 0 else "reduction_max_ops"] = amount
        elif operation.kind in ("copy", "async_copy", "fill"):
            work["elementwise_ops"] = operation_work(operation)["elementwise_ops"]
        else:
            unknown.append(f"operation {operation.index}: unsupported compute operation {operation.kind}")
        for kind, amount in work.items():
            if amount == 0:
                continue
            if visits is None or amount is None:
                totals[kind] = None
                unknown.append(f"operation {operation.index}: unresolved {kind} or loop visits")
            elif totals[kind] is not None:
                totals[kind] += amount * visits
        operations.append(dict(operation=operation.index, kind=operation.kind, work=work, visits=visits))
    return {
        "work_per_cta": totals,
        "matrix_signatures": signatures,
        "reduction_dtypes": sorted(reduction_dtypes),
        "target_arch": pressure.get("target_arch"),
        "operations": operations,
        "omitted_non_arithmetic_operations": sorted(omitted),
        "precision": "unknown" if unknown else "estimate",
        "unknown": sorted(set(unknown)),
        "assumptions": [
            "logical value-tree arithmetic excludes buffer-address expressions; moves, casts and initialization count as scalar work",
            "parallel extents and serial loop visits are multiplied exactly once; predicated branches count their full requested work",
            "scalar ALU operations have unit logical weight; exp and rsqrt use separate fixed primitive rates",
            "reductions count source minus destination elements, plus destination combines when clear=False",
            "native matrix instruction selection only; no layout inference, kernel-family recognition or pipeline timing",
            "TCGen05 uses the dtype-matched primitive rate; block-scale transfer/transpose and two-CTA synchronization costs are not modeled",
        ],
    }
