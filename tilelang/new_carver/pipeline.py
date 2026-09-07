"""Tile pipeline work and a bounded-buffer overlap model.

No iteration unrolling and no timing fitted to candidate benchmarks. Timing is
available only with an explicit effective per-SM cost profile. Per-buffer ready
and reuse events model producer overlap. Instruction scheduling and CUDA
dispatch remain estimates.
"""

from math import isfinite, prod
from tilelang import tvm
from tvm import tirx as tir


RATE_FIELDS = {
    "global_bytes_per_cycle",
    "shared_bytes_per_cycle",
    "gemm_flops_per_cycle",
    "wgmma_flops_per_cycle_per_warpgroup",
    "latency_scale",
    "reference_clock_mhz",
    "elementwise_ops_per_cycle",
    "exp_ops_per_cycle",
    "reduction_ops_per_cycle",  # Legacy profile field; not used for mapped tile reductions.
    "reduction_local_sum_per_cycle",
    "reduction_local_max_per_cycle",
    "reduction_shuffle_sum_per_cycle",
    "reduction_shuffle_max_per_cycle",
}
LATENCY_FIELDS = {"copy_latency_cycles", "barrier_cycles"}
CONSUMER_RATE_FIELDS = {
    "elementwise_ops_per_cycle",
    "exp_ops_per_cycle",
    "reduction_local_sum_per_cycle",
    "reduction_local_max_per_cycle",
    "reduction_shuffle_sum_per_cycle",
    "reduction_shuffle_max_per_cycle",
}
PROFILE_METADATA_FIELDS = {"gemm_signature", "profile_target", "profile_id", "memory_regime", "reduction_dtype"}


def validate_performance_model(profile):
    if not isinstance(profile, dict) or set(profile) - RATE_FIELDS - LATENCY_FIELDS - PROFILE_METADATA_FIELDS - {"consumer_rates"}:
        raise ValueError("performance_model contains unsupported fields")
    for key, value in profile.items():
        if key == "consumer_rates":
            if not isinstance(value, dict) or not value:
                raise ValueError("consumer_rates requires measured thread-count rows")
            for threads, rates in value.items():
                if not isinstance(threads, str) or not threads.isdigit() or not 0 < int(threads) <= 1024 or int(threads) % 32:
                    raise ValueError("consumer_rates keys must be positive warp-multiple thread counts")
                if not isinstance(rates, dict) or not rates or set(rates) - CONSUMER_RATE_FIELDS:
                    raise ValueError("consumer_rates contains unsupported primitives")
                validate_performance_model(rates)
            continue
        if key in PROFILE_METADATA_FIELDS:
            if key == "gemm_signature":
                if (
                    not isinstance(value, dict)
                    or set(value) != {"instruction", "a_dtype", "b_dtype", "accum_dtype"}
                    or not all(isinstance(v, str) and v for v in value.values())
                ):
                    raise ValueError("gemm_signature requires instruction, a_dtype, b_dtype and accum_dtype strings")
            elif key == "memory_regime" and value not in ("cached", "streaming"):
                raise ValueError("memory_regime must be cached or streaming")
            elif not isinstance(value, str) or not value:
                raise ValueError(f"{key} must be a nonempty string")
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, float | int)
            or not isfinite(value)
            or value < 0
            or (key in RATE_FIELDS and value == 0)
        ):
            raise ValueError("performance_model requires finite positive rates and nonnegative latencies")


def operation_work(op):
    from .analysis import _int
    from .specializations import call_names

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


def compute_participants(op, pressure, pass_configs=None):
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
        effective = dict(PassContext.current().config)
        effective.update(pass_configs or {})
        registered = PassContext.list_configs()
        with PassContext(config={k: v for k, v in effective.items() if k in registered}):
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


def _consumer_threads(op):
    from .analysis import _int

    values = [_int(v) for k, v in op.launch_threads.items() if k.startswith("threadIdx.")]
    return prod(values) if values and all(v is not None and v > 0 for v in values) else None


def estimate_pipeline_cycles(pipeline, concurrent_ctas=1, *, iterations=None):
    """Shared throughput is divided among resident CTAs; latency is not."""
    profile = pipeline.get("performance_model")
    if not profile or pipeline.get("unknown"):
        return None
    n = pipeline["iterations"]["max"] if iterations is None else iterations
    stages = pipeline["effective_buffer_depth"]
    global_rate = profile.get("global_bytes_per_cycle")
    latency = profile.get("copy_latency_cycles")
    barrier = profile.get("barrier_cycles")
    if not global_rate or latency is None or barrier is None or n is None or stages is None:
        return None

    def phase_time(phase):
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

    times = [phase_time(p) for p in pipeline["phases"]]
    if any(t is None for t in times):
        return None
    consumer = sum(t for t, p in zip(times, pipeline["phases"]) if p["inside_loop"])
    outside = sum(t for t, p in zip(times, pipeline["phases"]) if not p["inside_loop"])
    service = pipeline["input_bytes_per_iteration"] * concurrent_ctas / global_rate
    ready = service + pipeline["producer_copies_per_iteration"] * latency
    consumer += barrier * pipeline["producer_copies_per_iteration"]
    model = "serial consumer loop"
    if pipeline["overlap_eligible"]:
        from .tile_schedule import buffer_transition, repeat_transition

        copies = pipeline.get("producer_buffers")
        if not copies or pipeline.get("producer_schedule_unknown"):
            return None
        producer_ids = {copy["operation"] for copy in copies}
        consumers = [
            (p["operation"], t) for p, t in zip(pipeline["phases"], times) if p["inside_loop"] and p["operation"] not in producer_ids
        ]
        transition = buffer_transition(copies, consumers, stages, global_rate / concurrent_ctas, latency, barrier)
        ready = service + latency
        loop_cycles = repeat_transition(transition, n)[0]
        first = repeat_transition(transition, 1)[0]
        step = loop_cycles - repeat_transition(transition, n - 1)[0] if n else 0
        model = "per-buffer max-plus recurrence"
    else:
        step = ready + consumer
        first = step
        loop_cycles = n * step
    outside += pipeline["outside_loop_bytes"] * concurrent_ctas / global_rate + pipeline["outside_loop_input_copies"] * latency
    return {
        "cycles": outside + loop_cycles,
        "consumer_cycles_per_iteration": consumer,
        "copy_service_cycles_per_iteration": service,
        "input_ready_latency_cycles": ready,
        "steady_state_interval_cycles": step,
        "fill_and_first_consumer_cycles": first,
        "iterations": n,
        "schedule_model": model,
        "phase_cycles": [{"operation": p["operation"], "cycles": t} for p, t in zip(pipeline["phases"], times)],
        "outside_loop_cycles": outside,
        "concurrent_ctas": concurrent_ctas,
    }


def analyze_pipeline(col, specialization, memory, pressure, performance_model=None, pass_configs=None):
    from .analysis import _int
    from .memory import loop_visits
    from .specializations import in_loop

    loop = specialization.loop
    unknown = list(col.unknown)
    if performance_model and performance_model.get("profile_target", pressure.get("target_arch")) != pressure.get("target_arch"):
        unknown.append("device profile target does not match the analyzed kernel")
    phases = [
        {
            "operation": op.index,
            "phase": specialization.phase(op),
            "kind": op.kind,
            "dependencies": op.dependencies,
            "inside_loop": in_loop(op, loop),
            "work": operation_work(op),
            "compute_participants": compute_participants(op, pressure, pass_configs),
            "consumer_threads": _consumer_threads(op),
        }
        for op in col.operations
    ]
    from .reduction import reduction_work
    from .cta_work import collect_cta_work
    from .tile_schedule import collect_producer_buffers

    participants = {p["operation"]: p["compute_participants"] for p in phases}
    layout_cache = {}
    for op, phase in zip(col.operations, phases):
        phase["reduction"] = reduction_work(op, col, pressure, participants, layout_cache)
        reduction = phase["reduction"]
        if reduction and reduction["precision"] == "unknown":
            unknown.append(f"operation {op.index} reduction: {reduction.get('reason', 'unresolved mapping')}")
        elif reduction and performance_model:
            if reduction["dtype"] != performance_model.get("reduction_dtype", "float32"):
                unknown.append(f"operation {op.index} reduction dtype does not match the profile")
            for primitive in ("local", "shuffle"):
                rate = f"reduction_{primitive}_{reduction['operator']}_per_cycle"
                if reduction[f"{primitive}_pairs"] and not performance_model.get(rate):
                    unknown.append(f"operation {op.index} requires profile rate {rate}")
    distribution = collect_cta_work(col, loop)
    try:
        producer_buffers, producer_unknown = collect_producer_buffers(col, loop), []
    except Exception as error:
        producer_buffers, producer_unknown = [], [str(error)]
    if performance_model and performance_model.get("gemm_signature"):
        signature = performance_model["gemm_signature"]
        if any(
            p["work"]["gemm_flops"] and any((p["compute_participants"] or {}).get(k) != v for k, v in signature.items()) for p in phases
        ):
            unknown.append("device profile GEMM instruction/dtype signature does not match the kernel")
    iterations = {"min": None, "max": None, "precision": "unknown"}
    stages = None
    if loop is None:
        unknown.append("no recognized single tile pipeline loop")
    else:
        stages = _int(loop.annotations.get("num_stages", 0))
        representative = next(op for op in col.operations if in_loop(op, loop))
        domains = tuple(entry for entry in representative.loops if entry[2] == "4" or entry[0].same_as(loop.loop_var))
        iterations = loop_visits(domains)
        if iterations["max"] is None or stages is None:
            unknown.append("unresolved loop count or buffer depth")
    if any(
        kind not in ("4", "1") and (loop is None or not var.same_as(loop.loop_var)) for op in col.operations for var, _, kind in op.loops
    ):
        unknown.append("nested serial loop scheduling is not modeled")
    name = str(loop.loop_var) if loop is not None else None
    repeating = [tile for tile in memory["input_tiles"] if name in tile.get("loop_variables", [])]
    once = [tile for tile in memory["input_tiles"] if name not in tile.get("loop_variables", [])]
    if any(tile["tile_bytes"] is None or tile["bytes_per_block"] is None for tile in memory["input_tiles"] + memory["output_tiles"]):
        unknown.append("unresolved memory tile size")
    if any(value is None for phase in phases for value in phase["work"].values()):
        unknown.append("unresolved operation work")
    ws = pressure.get("warp_specialization", {})
    eligible = ws.get("status") == "predicted"
    if stages and not eligible:
        unknown.append("positive-stage pipeline scheduling policy is unresolved or unsupported")
    if any(op.predicates for op in col.operations):
        unknown.append("branch-dependent operation schedule")

    def known_sum(values):
        return sum(values) if all(value is not None for value in values) else None

    result = {
        "specialization": specialization.name,
        "phases": phases,
        "cta_work": distribution,
        "producer_buffers": producer_buffers,
        "producer_schedule_unknown": producer_unknown,
        "iterations": iterations,
        "num_stages": stages,
        "effective_buffer_depth": max(1, stages) if stages is not None else None,
        "overlap_eligible": eligible,
        "producer_copies_per_iteration": len(repeating),
        "input_bytes_per_iteration": known_sum([t["tile_bytes"] for t in repeating]),
        "outside_loop_input_copies": len(once),
        "outside_loop_bytes": known_sum([t["bytes_per_block"] for t in once] + [memory["output_bytes_per_block"]]),
        "loop_carried_buffers": (pressure.get("tile_liveness") or {}).get("loop_carried_buffers", []),
        "performance_model": performance_model,
        "unknown": sorted(set(unknown)),
        "precision": "unknown" if unknown else "conservative" if iterations["precision"] != "exact" else "estimate",
        "assumptions": [
            "each producer tile is waited on at first use and released after its last consumer",
            "max-plus recurrence models separate buffer rings and shared byte-service capacity",
            "a finite buffer ring limits overlap; loop-carried consumer state serializes consumer iterations",
            "no loop unrolling; startup and repeated steady-state intervals are modeled analytically",
            "CTA timing reports maximum work; grid ranking uses the separately collected per-CTA work distribution",
            "explicit profile rates are effective per-SM rates for the target and operation dtypes; no compiler counters or benchmark latencies",
            "optional WGMMA per-warpgroup rate limits a CTA's compute service independently of the aggregate per-SM rate",
            "optional single-CTA primitive rates limit scalar/exp/reduction service at the original consumer thread count",
            "consumer probes use eight independent chains; actual instruction dependencies and phase occupancy can differ",
        ],
    }
    result["timing"] = estimate_pipeline_cycles(result)
    result["timing_status"] = "estimate" if result["timing"] is not None else "unknown"
    return result
