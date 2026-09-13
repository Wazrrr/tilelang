"""Operation work, fragment ownership and effective service cycles."""

from collections import Counter, OrderedDict
from itertools import product
from math import prod
import operator
from tilelang import tvm
from tvm import tirx as tir
from .profiling.profile_schema import CONSUMER_RATE_FIELDS
from .src.ir_utils import _int, call_names

_CACHE = OrderedDict()


def operation_work(op):
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
    from tilelang.transform import PassContext
    from tvm.target import Target

    if not hasattr(op.metadata, "cRegion"):
        return None
    threads = [_int(v) for k, v in op.launch_threads.items() if k.startswith("threadIdx.")]
    if not threads or any(v is None or v <= 0 for v in threads) or not pressure.get("target_arch"):
        return {"precision": "unknown"}
    try:
        from .targets import resolve_target

        model = pressure.get("target_model")
        target_description = (
            {"kind": model["kind"], "mcpu" if model["kind"] == "hip" else "arch": model["arch"]}
            if model
            else {"kind": "cuda", "arch": pressure["target_arch"]}
        )
        target = Target(resolve_target(target_description).compiler_target())
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
    values = [_int(v) for k, v in op.launch_threads.items() if k.startswith("threadIdx.")]
    return prod(values) if values and all(v is not None and v > 0 for v in values) else None


def _evaluator(expr, variables):
    """Compile a small, read-only integer mapping into Python closures."""
    if isinstance(expr, tir.IntImm):
        value = int(expr)
        return lambda _: value
    if isinstance(expr, tir.Var):
        index = variables.index(expr)
        return lambda values: values[index]
    if isinstance(expr, tir.Cast):
        return _evaluator(expr.value, variables)
    operations = {
        tir.Add: operator.add,
        tir.Sub: operator.sub,
        tir.Mul: operator.mul,
        tir.FloorDiv: operator.floordiv,
        tir.FloorMod: operator.mod,
        tir.Min: min,
        tir.Max: max,
    }
    for kind, function in operations.items():
        if isinstance(expr, kind):
            left, right = _evaluator(expr.a, variables), _evaluator(expr.b, variables)
            return lambda values: function(left(values), right(values))
    raise ValueError("unsupported fragment thread expression")


def fragment_reduction_work(layout, shape, axis, subgroup_size=32):
    """Count physical local pairs and butterfly lane pairs, including replication.

    Enumerates a bounded logical tile, never the enclosing loop or launch grid.
    Inter-warp collectives remain unknown until their communication is modeled.
    """

    if subgroup_size not in (32, 64):
        raise ValueError("reduction ownership requires a modeled subgroup size (32 or 64)")
    replication = _int(layout.replicate_size)
    if not replication or prod(shape) * replication > 262144:
        raise ValueError("unresolved or oversized fragment ownership map")
    forward_vars = list(layout.get_forward_vars())
    if len(forward_vars) not in (len(shape), len(shape) + 1):
        raise ValueError("fragment rank does not match reduction source")
    # FragmentNode::GetForwardVars prepends the replication coordinate.
    variables = forward_vars[-len(shape) :]
    key = (str(layout.thread), tuple(str(v) for v in variables), tuple(shape), axis, replication, subgroup_size)
    if key in _CACHE:
        return dict(_CACHE[key])
    used = set()
    tir.stmt_functor.post_order_visit(layout.thread, lambda n: used.add(n) if isinstance(n, tir.Var) else None)
    extra = [v for v in used if not any(v.same_as(x) for x in variables)]
    if len(extra) > 1 or (replication > 1 and not extra):
        raise ValueError("unresolved fragment replication")
    evaluate = _evaluator(layout.thread, variables + extra)
    local = shuffle = 0
    widths, local_sizes = set(), set()
    other_axes = [i for i in range(len(shape)) if i != axis]
    for replica in range(replication):
        for row in product(*(range(shape[i]) for i in other_axes)):
            values = [0] * len(shape)
            for i, coordinate in zip(other_axes, row):
                values[i] = coordinate
            owners = Counter()
            for position in range(shape[axis]):
                values[axis] = position
                owners[evaluate(values + ([replica] if extra else []))] += 1
            width = len(owners)
            if width & (width - 1) or len({thread // subgroup_size for thread in owners}) != 1:
                raise ValueError("reduction needs an unresolved or inter-warp collective")
            # Match the compiler's XOR butterfly over a power-of-two lane set.
            first = min(owners)
            xor = {thread ^ first for thread in owners}
            bits = [1 << b for b in range(subgroup_size.bit_length() - 1) if (1 << b) in xor]
            if len(xor) != 1 << len(bits) or any(value & ~sum(bits) for value in xor):
                raise ValueError("reduction lanes are not a supported butterfly group")
            local += sum(n - 1 for n in owners.values())
            shuffle += width * (width.bit_length() - 1)
            widths.add(width)
            local_sizes.update(owners.values())
    result = dict(
        local_pairs=local,
        shuffle_pairs=shuffle,
        lane_widths=sorted(widths),
        values_per_lane=sorted(local_sizes),
        replication=replication,
        precision="predicted",
        assumptions=["compiler fragment ownership; local accumulation followed by a butterfly within one warp"],
    )
    _CACHE[key] = result
    if len(_CACHE) > 256:
        _CACHE.popitem(last=False)
    return dict(result)


def reduction_work(op, col, pressure, participants, layout_cache):
    """Use explicit ownership or the compiler's read-only GEMM layout helper."""

    meta = op.metadata
    if not hasattr(meta, "dim") or not hasattr(meta, "srcRegion"):
        return None
    result = {"precision": "unknown", "dtype": str(meta.src.dtype)}
    try:
        kind = int(meta.type.type)
        if kind not in (0, 2):
            raise ValueError("only sum and max reduction rates are modeled")
        result["operator"] = "sum" if kind == 0 else "max"
        shape = [_int(x.extent) for x in meta.srcRegion.region]
        axis = _int(meta.dim)
        if axis is None or not all(n is not None and n > 0 for n in shape):
            raise ValueError("unresolved reduction tile")
        if any(_int(r.min) != 0 or _int(r.extent) != _int(s) for r, s in zip(meta.srcRegion.region, meta.src.shape)):
            raise ValueError("partial reduction fragment requires a region mapping")
        layout = col.layouts.get(meta.src.data)
        source = "explicit fragment layout"
        if layout is None:
            if meta.src not in layout_cache:
                producers = [p for p in col.operations[: op.index] if hasattr(p.metadata, "cRegion") and p.metadata.c.same_as(meta.src)]
                if not producers:
                    raise ValueError("no known fragment producer for reduction")
                producer = producers[-1]
                compute = participants.get(producer.index) or {}
                instruction = compute.get("instruction")
                if instruction not in ("cuda.wgmma", "cuda.mma"):
                    raise ValueError("automatic reduction mapping requires an MMA or WGMMA fragment")
                # This helper creates layout values; it runs no lowering pass,
                # assigns no IR annotation, and does not change the operator.
                from tilelang.cuda.op.gemm.gemm_wgmma import GemmWGMMA
                from tilelang.cuda.op.gemm.gemm_mma import GemmMMA
                from tvm.target import Target

                helper = GemmWGMMA if instruction == "cuda.wgmma" else GemmMMA
                layouts = helper(producer.metadata).infer_layout(
                    Target({"kind": "cuda", "arch": pressure["target_arch"]}), compute["consumer_threads"]
                )
                layout_cache[meta.src] = layouts[meta.src]
            layout = layout_cache[meta.src]
            source = "compiler MMA/WGMMA fragment layout prediction"
        subgroup = pressure.get("target_model", {}).get("subgroup_size", 32)
        if subgroup is None and pressure.get("target_model", {}).get("kind") is None:
            subgroup = 32  # Preserve the original no-target fragment contract.
        result.update(fragment_reduction_work(layout, shape, axis, subgroup), mapping_source=source)
    except Exception as error:
        result["reason"] = str(error)
    return result


def estimate_phase_cycles(phase, profile, concurrent_ctas):
    """Apply aggregate SM rates and optional consumer/warpgroup ceilings."""
    terms = {}

    def service(amount, rate_key):
        if not amount:
            return 0
        rate = profile.get(rate_key)
        if not rate:
            return None
        aggregate = amount * concurrent_ctas / rate
        if rate_key not in CONSUMER_RATE_FIELDS or not profile.get("consumer_rates"):
            return aggregate
        # A CTA cannot use all SM issue capacity when too few consumer
        # warps are ready. Producer warps do not execute these operations.
        row = profile["consumer_rates"].get(str(phase.get("consumer_threads")), {})
        single = row.get(rate_key)
        return max(aggregate, amount / single) if single else None

    for key, amount in phase["work"].items():
        if key == "reduction_ops":
            terms[key] = 0
            continue
        if amount is None:
            return None
        rate_key = {
            "gemm_flops": "gemm_flops_per_cycle",
            "shared_bytes": "shared_bytes_per_cycle",
            "elementwise_ops": "elementwise_ops_per_cycle",
            "exp_ops": "exp_ops_per_cycle",
            "reduction_ops": "reduction_ops_per_cycle",
        }[key]
        if amount and not profile.get(rate_key):
            return None
        terms[key] = service(amount, rate_key)
        if terms[key] is None:
            return None
    if phase["work"]["reduction_ops"]:
        reduction = phase.get("reduction") or {}
        if reduction.get("precision") != "predicted" or reduction["dtype"] != profile.get("reduction_dtype", "float32"):
            return None
        kind = reduction["operator"]
        for operation in ("local", "shuffle"):
            amount = reduction[f"{operation}_pairs"]
            rate = profile.get(f"reduction_{operation}_{kind}_per_cycle")
            if amount and not rate:
                return None
            cycles = service(amount, f"reduction_{operation}_{kind}_per_cycle")
            if cycles is None:
                return None
            terms["reduction_ops"] += cycles
    # Matrix instructions consume tensor-core and shared-memory service;
    # scalar/reduction/exp phases execute in program order.
    group_service = 0
    group_rate = profile.get("wgmma_flops_per_cycle_per_warpgroup")
    if group_rate and phase["work"]["gemm_flops"]:
        participants = phase.get("compute_participants") or {}
        if participants.get("precision") != "predicted":
            return None
        if participants["instruction"] == "cuda.wgmma":
            groups = participants.get("warpgroups")
            if not groups:
                return None
            group_service = phase["work"]["gemm_flops"] / (groups * group_rate)
    return (
        max(terms["gemm_flops"], terms["shared_bytes"], group_service)
        + terms["elementwise_ops"]
        + terms["exp_ops"]
        + terms["reduction_ops"]
    )
