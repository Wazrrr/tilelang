"""Portable numerical pipeline evaluation, preserving the CUDA equations."""

from .compute import estimate_phase_cycles


def estimate_pipeline_cycles(pipeline, concurrent_ctas=1, *, iterations=None):
    """Shared throughput is divided among resident CTAs; latency is not."""
    if "region_schedule" in pipeline:
        from .region_schedule import estimate_region_cycles

        return estimate_region_cycles(pipeline, concurrent_ctas, iterations=iterations)
    profile = pipeline.get("performance_model")
    if not profile or pipeline.get("unknown"):
        return None
    n = pipeline["iterations"]["max"] if iterations is None else iterations
    stages = pipeline["effective_buffer_depth"]
    global_rate = profile.get("global_bytes_per_cycle")
    latency = profile.get("copy_latency_cycles")
    barrier = profile.get("barrier_cycles")
    if not global_rate or barrier is None or n is None or stages is None:
        return None

    times = [estimate_phase_cycles(p, profile, concurrent_ctas) for p in pipeline["phases"]]
    if any(t is None for t in times):
        return None
    if pipeline.get("ampere_schedule") is not None:
        if latency is None:
            return None
        return _estimate_ampere(pipeline, times, n, profile, concurrent_ctas)
    consumer = sum(t for t, p in zip(times, pipeline["phases"]) if p["inside_loop"])
    outside = sum(t for t, p in zip(times, pipeline["phases"]) if not p["inside_loop"])
    service_bytes = pipeline.get("input_service_bytes_per_iteration") or pipeline["input_bytes_per_iteration"]
    service = service_bytes * concurrent_ctas / global_rate
    copies_per_iteration = pipeline["producer_copies_per_iteration"]
    ready = service
    model = "synchronous copy/consumer loop"
    if pipeline["overlap_eligible"]:
        if latency is None:
            return None
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
        # Stage-zero lowering emits direct cooperative loads, then one CTA
        # barrier before shared-memory consumption and one before the next
        # iteration reuses the buffers. Logical source buffers share those
        # synchronization points; they are not independent TMA transactions.
        consumer += (2 * barrier) if copies_per_iteration else 0
        step = ready + consumer
        first = step
        loop_cycles = n * step
    outside += pipeline["outside_loop_bytes"] * concurrent_ctas / global_rate
    if pipeline["outside_loop_input_copies"]:
        if pipeline["overlap_eligible"]:
            outside += pipeline["outside_loop_input_copies"] * latency
        else:
            # Direct outside-loop copies likewise become visible through CTA
            # synchronization, not the asynchronous-copy readiness latency.
            outside += 2 * barrier
    return {
        "cycles": outside + loop_cycles,
        "consumer_cycles_per_iteration": consumer,
        "copy_service_cycles_per_iteration": service,
        "logical_input_bytes_per_iteration": pipeline["input_bytes_per_iteration"],
        "service_bytes_per_iteration": service_bytes,
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
