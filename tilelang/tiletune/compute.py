"""Operation work, fragment ownership and effective service cycles."""

from collections import Counter, OrderedDict
from copy import deepcopy
from itertools import product
from math import prod
import operator
from tilelang import tvm
from tvm import tirx as tir
from .src.ir_utils import _int, call_names

_CACHE = OrderedDict()


def operation_work(op, col=None):
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
    elif op.kind == "reduce" and hasattr(meta, "srcRegion"):
        src, dst = size(meta.srcRegion), size(meta.dstRegion)
        work["reduction_ops"] = src - dst if src is not None and dst is not None else None
    elif op.kind == "elementwise":
        # Parallel loops define one logical tile, not work per launch thread.
        # The collector serializes the native integer kind, whereas the Python
        # enum's string representation is "ForKind.PARALLEL".
        dims = [_int(r.extent) for _, r, kind in op.loops if kind == "1"]
        elements = prod(dims) if all(v is not None for v in dims) else None
        names = call_names(op)
        exps = sum(name in ("tirx.exp", "tirx.exp2") for name in names)
        scalar_ops = []
        types = (tir.Add, tir.Sub, tir.Mul, tir.Div, tir.FloorDiv, tir.Max, tir.Min, tir.Cast, tir.Select)
        tir.stmt_functor.post_order_visit(op.metadata.value, lambda n: scalar_ops.append(n) if isinstance(n, types) else None)
        work["exp_ops"] = elements * exps if elements is not None else None
        rsqrts = sum(name == "tirx.rsqrt" for name in names)
        if rsqrts:
            work["rsqrt_ops"] = elements * rsqrts if elements is not None else None
        logs = sum(name == "tirx.log2" for name in names)
        if logs:
            work["log_ops"] = elements * logs if elements is not None else None
        work["elementwise_ops"] = elements * max(len(scalar_ops), 1) if elements is not None else None
        if any(
            name not in ("tirx.exp", "tirx.exp2", "tirx.rsqrt", "tirx.log2", "tirx.if_then_else", "tirx.likely", "tl.infinity")
            for name in names
        ):
            work["elementwise_ops"] = None
        elif col is not None and getattr(col, "inferred_layouts", {}).get(op.metadata.buffer.data) is not None:
            try:
                work.update(scalar_fragment_work(op, col.inferred_layouts[op.metadata.buffer.data]))
            except Exception as error:
                work["elementwise_ops"] = None
                col.scalar_work_unknown = getattr(col, "scalar_work_unknown", {})
                col.scalar_work_unknown[op.index] = str(error)
    elif op.kind in ("copy", "async_copy", "fill") and op.writes:
        # Register casts, initialization and epilogue stores are consumer work.
        # External read copies are charged to the producer separately.
        if not any(r.buffer.scope() == "global" for r in op.reads):
            dims = [_int(r.extent) for r in op.writes[0].ranges]
            work["elementwise_ops"] = prod(dims) if all(v is not None for v in dims) else None
            if len(op.reads) == 1 and str(op.reads[0].buffer.dtype) == "float32":
                dtype = str(op.writes[0].buffer.dtype)
                if dtype in ("float8_e4m3fn", "float8_e5m2"):
                    # CUDA lowers the example's epilogue to packed FP8
                    # conversion instructions. Charge converted values to a
                    # measured dtype-specific rate, not FP32 arithmetic.
                    work[f"convert_float32_to_{dtype}"] = work["elementwise_ops"]
                    work["elementwise_ops"] = 0
    else:
        # Access-region reflection is not a timing implementation. In particular
        # scans, transpose and atomics must not silently receive zero cost.
        work["elementwise_ops"] = None
    return work


def _regular_projection_counter(op, layout, axes):
    """Exact cardinalities for disjoint thread bit fields, in O(sum(axis sizes)).

    This proves separability from the expression tree and checks disjoint bit
    fields over each complete axis domain. It never guesses from a few points.
    Arbitrary coupled mappings fall back to the bounded reference enumerator.
    """
    variables = [v for v, _, _ in axes]
    forward = list(layout.get_forward_vars())[-len(op.metadata.indices) :]
    expression = tir.stmt_functor.substitute(layout.thread, dict(zip(forward, op.metadata.indices)))
    used = set()
    tir.stmt_functor.post_order_visit(expression, lambda n: used.add(n) if isinstance(n, tir.Var) else None)
    extra = [v for v in used if v not in variables]
    layout_variables = set()
    tir.stmt_functor.post_order_visit(layout.thread, lambda n: layout_variables.add(n) if isinstance(n, tir.Var) else None)
    if any(v not in layout_variables or v in forward for v in extra):
        raise ValueError("scalar indices depend on a nonparallel coordinate")
    if len(extra) > 1:
        raise ValueError("coupled or unresolved scalar ownership")
    dimensions = list(axes) + [(v, 0, _int(layout.replicate_size)) for v in extra]
    terms = {v: [] for v, _, _ in dimensions}
    constants = []

    def split(expr):
        if isinstance(expr, tir.Add):
            split(expr.a)
            split(expr.b)
            return
        dependencies = set()
        tir.stmt_functor.post_order_visit(expr, lambda n: dependencies.add(n) if isinstance(n, tir.Var) else None)
        if not dependencies:
            constants.append(_int(expr))
        elif len(dependencies) == 1 and next(iter(dependencies)) in terms:
            terms[next(iter(dependencies))].append(expr)
        else:
            raise ValueError("thread expression couples multiple scalar axes")

    split(expression)
    if any(n is None or n < 0 for n in constants):
        raise ValueError("unresolved thread offset")
    occupied = sum(constants)
    counts = []
    for var, lo, n in dimensions:
        evaluate = _evaluator(sum(terms[var], tir.IntImm(var.dtype, 0)), [var])
        values = {evaluate([i]) for i in range(lo, lo + n)}
        bits = 0
        for value in values:
            if value < 0:
                raise ValueError("negative thread contribution")
            bits |= value
        if bits & occupied:
            raise ValueError("thread contributions overlap")
        occupied |= bits
        counts.append(len(values))

    def count(mask):
        return prod(n if i in mask else counts[i] for i, (_, _, n) in enumerate(dimensions))

    return count


def scalar_fragment_work(op, layout, *, reference=False):
    """Count per-thread expression values, including replication and local CSE.

    Only pure scalar expressions with compiler-inferred ownership enter this
    path. Buffer address arithmetic is not counted as FP32 arithmetic. Distinct
    parallel coordinates needed by each expression determine which evaluations
    can be shared within a thread; no sharing is assumed between threads.
    """
    axes = [(var, _int(r.min), _int(r.extent)) for var, r, kind in op.loops if kind == "1"]
    replication = _int(layout.replicate_size)
    if not replication or any(lo is None or n is None for _, lo, n in axes) or prod(n for _, _, n in axes) * replication > 262144:
        raise ValueError("unresolved or oversized scalar ownership map")
    variables = [var for var, _, _ in axes]
    shape = list(op.metadata.buffer.shape)
    forward = list(layout.get_forward_vars())[-len(shape) :]
    used = set()
    tir.stmt_functor.post_order_visit(layout.thread, lambda n: used.add(n) if isinstance(n, tir.Var) else None)
    extra = [var for var in used if not any(var.same_as(v) for v in forward)]
    if len(extra) > 1:
        raise ValueError("unresolved scalar layout replication")
    try:
        if reference:
            raise ValueError("bounded reference requested")
        count = _regular_projection_counter(op, layout, axes)
    except ValueError:
        owner = _evaluator(layout.thread, forward + extra)
        indices = [_evaluator(index, variables) for index in op.metadata.indices]
        points = [
            (owner([index(coords) for index in indices] + ([replica] if extra else [])), coords)
            for coords in product(*(range(lo, lo + n) for _, lo, n in axes))
            for replica in range(replication)
        ]

        def count(mask):
            return len({(thread, *(coords[i] for i in mask)) for thread, coords in points})

    counts = {}

    def amount(node):
        used = set()
        tir.stmt_functor.post_order_visit(node, lambda n: used.add(n) if isinstance(n, tir.Var) else None)
        mask = tuple(i for i, var in enumerate(variables) if var in used)
        if mask not in counts:
            counts[mask] = count(mask)
        return counts[mask]

    result = dict(elementwise_ops=0, exp_ops=0)
    visited = {}
    binary = (tir.Add, tir.Sub, tir.Mul, tir.Div, tir.FloorDiv, tir.Max, tir.Min)

    def visit(node):
        key = tvm.ir.structural_hash(node)
        if any(tvm.ir.structural_equal(node, previous) for previous in visited.get(key, [])):
            return
        visited.setdefault(key, []).append(node)
        if isinstance(node, tir.BufferLoad):
            return
        if isinstance(node, binary):
            result["elementwise_ops"] += amount(node)
            visit(node.a)
            visit(node.b)
        elif isinstance(node, tir.Cast):
            result["elementwise_ops"] += amount(node)
            visit(node.value)
        elif isinstance(node, tir.Select):
            result["elementwise_ops"] += amount(node)
            visit(node.true_value)
            visit(node.false_value)
        elif isinstance(node, tir.Call):
            name = node.op.name if hasattr(node.op, "name") else str(node.op)
            field = {"tirx.exp": "exp_ops", "tirx.exp2": "exp_ops", "tirx.rsqrt": "rsqrt_ops", "tirx.log2": "log_ops"}.get(name)
            if field:
                result[field] = result.get(field, 0) + amount(node)
            elif name == "tirx.if_then_else":
                result["elementwise_ops"] += amount(node)
            for arg in node.args:
                visit(arg)

    visit(op.metadata.value)
    result["elementwise_ops"] = max(result["elementwise_ops"], count(tuple(range(len(axes)))))
    return result


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


def fragment_reduction_work(layout, shape, axis, subgroup_size=32, *, allow_interwarp=False):
    """Count physical local pairs and butterfly lane pairs, including replication.

    Enumerates a bounded logical tile, never the enclosing loop or launch grid.
    Compiler-inferred Ampere layouts also support shared-memory XOR rounds.
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
    key = (str(layout.thread), tuple(str(v) for v in variables), tuple(shape), axis, replication, subgroup_size, allow_interwarp)
    if key in _CACHE:
        return deepcopy(_CACHE[key])
    used = set()
    tir.stmt_functor.post_order_visit(layout.thread, lambda n: used.add(n) if isinstance(n, tir.Var) else None)
    extra = [v for v in used if not any(v.same_as(x) for x in variables)]
    if len(extra) > 1 or (replication > 1 and not extra):
        raise ValueError("unresolved fragment replication")
    evaluate = _evaluator(layout.thread, variables + extra)
    local = shuffle = shared = 0
    thread_rounds, thread_channels = Counter(), Counter()
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
            if width & (width - 1) or (not allow_interwarp and len({thread // subgroup_size for thread in owners}) != 1):
                raise ValueError("reduction needs an unresolved or inter-warp collective")
            # Match the compiler's XOR butterfly over a power-of-two lane set.
            first = min(owners)
            xor = {thread ^ first for thread in owners}
            bits = [1 << b for b in range(max(subgroup_size, max(owners) + 1).bit_length()) if (1 << b) in xor]
            if len(xor) != 1 << len(bits) or any(value & ~sum(bits) for value in xor):
                raise ValueError("reduction lanes are not a supported butterfly group")
            local += sum(n - 1 for n in owners.values())
            shuffle += width * sum(bit < subgroup_size for bit in bits)
            shared += width * sum(bit >= subgroup_size for bit in bits)
            rounds = 2 * sum(bit >= subgroup_size for bit in bits)
            if rounds:
                for thread in owners:
                    thread_rounds[thread] += rounds
                    thread_channels[thread] += 1
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
    if shared:
        result.update(
            shared_pairs=shared,
            barrier_rounds=max(thread_rounds.values()),
            workspace_reuse_barriers=max(n if n > 1 else 0 for n in thread_channels.values()),
            local_outputs_per_thread=max(thread_channels.values()),
        )
        result["assumptions"] = [
            "compiler fragment ownership; scalar AllReduce calls serialize local output channels",
            "each inter-warp XOR round has two barriers; repeated workspace use adds a fence per local channel",
        ]
    _CACHE[key] = result
    if len(_CACHE) > 256:
        _CACHE.popitem(last=False)
    return deepcopy(result)


def reduction_work(op, col, pressure, participants, layout_cache):
    """Use explicit ownership or the compiler's read-only GEMM layout helper."""

    meta = op.metadata
    if op.kind != "reduce" or not hasattr(meta, "srcRegion"):
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
        if getattr(col, "ampere_layout_unknown", None):
            raise ValueError(f"ownership failed compiler validation: {col.ampere_layout_unknown}")
        inferred = False
        verified = getattr(col, "inferred_layouts", {}).get(meta.src.data)
        if layout is not None and verified is not None:
            if not tvm.ir.structural_equal(layout, verified):
                raise ValueError("explicit ownership disagrees with compiler layout")
            source, inferred = "explicit fragment ownership verified by compiler LayoutInference", True
        if layout is None:
            layout = getattr(col, "inferred_layouts", {}).get(meta.src.data)
            if layout is not None:
                source, inferred = "compiler LayoutInference on an isolated Ampere IRModule", True
        if layout is None:
            if meta.src not in layout_cache:
                producers = [p for p in col.operations[: op.index] if hasattr(p.metadata, "cRegion") and p.metadata.c.same_as(meta.src)]
                if not producers:
                    if getattr(col, "ampere_layout_unknown", None):
                        raise ValueError(f"Ampere layout inference failed: {col.ampere_layout_unknown}")
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
        if inferred:
            if pressure.get("target_arch") not in ("sm_80", "sm_86", "sm_89"):
                raise ValueError("inter-warp collective verification requires the Ampere CUDA lowering")
            if _int(layout.get_thread_size()) != consumer_threads(op):
                raise ValueError("partial thread-domain collective is not modeled")
        subgroup = pressure.get("target_model", {}).get("subgroup_size", 32)
        if subgroup is None and pressure.get("target_model", {}).get("kind") is None:
            subgroup = 32  # Preserve the original no-target fragment contract.
        result.update(fragment_reduction_work(layout, shape, axis, subgroup, allow_interwarp=inferred), mapping_source=source)
        if result.get("shared_pairs"):
            if _int(meta.batch) != 1:
                raise ValueError("batched inter-warp reduction scheduling is not modeled")
            dtype = tvm.DataType(meta.src.dtype)
            result["workspace_bytes"] = consumer_threads(op) * dtype.bits * dtype.lanes // 8
    except Exception as error:
        result["precision"] = "unknown"
        result["reason"] = str(error)
    return result


from tiletune_core.compute import estimate_phase_cycles as estimate_phase_cycles
