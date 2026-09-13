"""Tile pipeline work and a bounded-buffer overlap model.

No iteration unrolling and no timing fitted to candidate benchmarks. Timing is
available only with an explicit effective per-SM cost profile. Per-buffer ready
and reuse events model producer overlap. Instruction scheduling and CUDA
dispatch remain estimates.
"""

from .compute import compute_participants, consumer_threads, operation_work
from .compute import estimate_phase_cycles


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

    times = [estimate_phase_cycles(p, profile, concurrent_ctas) for p in pipeline["phases"]]
    if any(t is None for t in times):
        return None
    if pipeline.get("ampere_schedule") is not None:
        return _estimate_ampere(pipeline, times, n, profile, concurrent_ctas)
    consumer = sum(t for t, p in zip(times, pipeline["phases"]) if p["inside_loop"])
    outside = sum(t for t, p in zip(times, pipeline["phases"]) if not p["inside_loop"])
    service = pipeline["input_bytes_per_iteration"] * concurrent_ctas / global_rate
    ready = service + pipeline["producer_copies_per_iteration"] * latency
    consumer += barrier * pipeline["producer_copies_per_iteration"]
    model = "serial consumer loop"
    if pipeline["overlap_eligible"]:
        from .schedule import buffer_transition, repeat_transition

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


def _estimate_ampere(pipeline, times, n, profile, concurrent_ctas):
    from .ampere import schedule_cycles

    phases = pipeline["phases"]
    copies = pipeline["producer_buffers"]
    copy_ids = {copy["operation"] for copy in copies}
    rate = profile["global_bytes_per_cycle"] / concurrent_ctas
    cost = {}
    for phase, cycles in zip(phases, times):
        work = phase["external_work"]
        if phase["inside_loop"] and phase["operation"] in copy_ids:
            cost[phase["operation"]] = cycles
        else:
            cost[phase["operation"]] = cycles + (work["read_bytes"] + work["write_bytes"]) / rate
            cost[phase["operation"]] += work["read_groups"] * profile["copy_latency_cycles"]
    outside = sum(cost[p["operation"]] for p in phases if not p["inside_loop"])
    consumer = sum(cost[p["operation"]] for p in phases if p["inside_loop"] and p["operation"] not in copy_ids)
    service = sum(copy["bytes"] for copy in copies) / rate
    plan = pipeline["ampere_schedule"]
    if pipeline["num_stages"]:
        if pipeline["producer_schedule_unknown"]:
            return None
        values = [schedule_cycles(plan, copies, cost, k, profile, concurrent_ctas) for k in (n, max(0, n - 1), 1)]
        if any(value is None for value in values):
            return None
        total, previous, first = values
        step = total - previous if n else 0
        model = "Ampere compiler-ordered asynchronous copy recurrence"
    else:
        first = consumer + service + len(copies) * (profile["copy_latency_cycles"] + profile["barrier_cycles"])
        step, total = first, n * first
        model = "Ampere serial per-operation memory and consumer schedule"
    return dict(
        cycles=outside + total,
        consumer_cycles_per_iteration=consumer,
        copy_service_cycles_per_iteration=service,
        input_ready_latency_cycles=service + len(copies) * profile["copy_latency_cycles"],
        steady_state_interval_cycles=step,
        fill_and_first_consumer_cycles=first,
        iterations=n,
        schedule_model=model,
        phase_cycles=[dict(operation=p["operation"], cycles=cost[p["operation"]]) for p in phases],
        outside_loop_cycles=outside,
        concurrent_ctas=concurrent_ctas,
    )


def analyze_pipeline(col, memory, pressure, performance_model=None, pass_configs=None, *, loop, family_name, phase_labels):
    from .src.ir_utils import _int
    from .src.ir_utils import loop_visits
    from .src.ir_utils import in_loop

    unknown = list(col.unknown)
    ampere = getattr(col, "ampere_plan", None)
    model = pressure.get("target_model")
    if model and model["kind"] is not None and not model["block_execution"]:
        unknown.append("target requires its own core/storage scheduling model")
    if (
        model
        and model["kind"] not in (None, "cuda")
        and performance_model
        and (performance_model.get("profile_backend") != model["kind"] or performance_model.get("profile_target") != model["arch"])
    ):
        unknown.append("non-CUDA timing requires an explicit matching profile_backend and profile_target")
    if ampere is None and any(
        op.kind == "elementwise" and in_loop(op, loop) and any(r.buffer.scope() == "global" for r in op.reads) for op in col.operations
    ):
        unknown.append("direct scalar global accesses inside a recurrence require a per-iteration access schedule")
    if performance_model and performance_model.get("profile_target", pressure.get("target_arch")) != pressure.get("target_arch"):
        unknown.append("device profile target does not match the analyzed kernel")
    phases = [
        {
            "operation": op.index,
            "phase": phase_labels[op.index],
            "kind": op.kind,
            "dependencies": op.dependencies,
            "inside_loop": in_loop(op, loop),
            "work": operation_work(op, col),
            "compute_participants": compute_participants(op, pressure, pass_configs),
            "consumer_threads": consumer_threads(op),
        }
        for op in col.operations
    ]
    from .compute import reduction_work
    from .schedule import collect_cta_work
    from .schedule import collect_producer_buffers

    participants = {p["operation"]: p["compute_participants"] for p in phases}
    layout_cache = {}
    for op, phase in zip(col.operations, phases):
        if op.index in getattr(col, "scalar_work_unknown", {}):
            unknown.append(f"operation {op.index} scalar ownership: {col.scalar_work_unknown[op.index]}")
        if op.kind == "elementwise" and getattr(col, "inferred_layouts", {}).get(op.metadata.buffer.data) is not None:
            phase["scalar_work_basis"] = (
                "compiler fragment ownership and pure-expression reuse within each thread; address instructions excluded"
            )
        if ampere is not None:
            from .ampere import external_work

            phase["external_work"] = external_work(op)
            if any(value is None for value in phase["external_work"].values()):
                unknown.append(f"operation {op.index} has unresolved external access bytes")
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
    if loop is None and not col.pipeline_loops and not col.serial_loops:
        # A single-pass tile has only outside-loop work. Zero recurrence steps
        # let the same timing equation count that work exactly once.
        stages = 0
        iterations = {"min": 0, "max": 0, "precision": "exact"}
    elif loop is None:
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
    if ampere is not None and stages:
        eligible = ampere["status"] == "predicted"
        unknown.extend(ampere["unknown"])
        if performance_model:
            for field in ("async_copy_issue_bytes_per_cycle", "async_copy_latency_cycles"):
                if field not in performance_model:
                    unknown.append(f"Ampere asynchronous pipeline requires profile field {field}")
    if stages and not eligible:
        unknown.append("positive-stage pipeline scheduling policy is unresolved or unsupported")
    if any(op.predicates for op in col.operations):
        unknown.append("branch-dependent operation schedule")

    def known_sum(values):
        return sum(values) if all(value is not None for value in values) else None

    result = {
        "specialization": family_name,
        "ampere_schedule": ampere,
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
