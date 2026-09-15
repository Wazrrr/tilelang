"""Numerical replay of resolved Ampere compiler schedules."""


def schedule_cycles(plan, copies, phase_cycles, iterations, profile, concurrent_ctas, *, runs=None):
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
    segments = []
    position = 0
    for run in runs or [dict(count=iterations, copies=copies, costs=phase_cycles)]:
        segments.append((position, position + run["count"], {c["operation"]: c for c in run["copies"]}, run["costs"]))
        position += run["count"]
    if position != iterations:
        return None
    boundaries = sorted(
        {
            0,
            iterations + plan["max_stage"],
            *(bound + event["stage"] for lo, hi, _, _ in segments for bound in (lo, hi) for event in events),
        }
    )
    for start, end in zip(boundaries, boundaries[1:]):
        warp, service = basis[:2]
        produced, waited = {}, set()
        for event in events:
            if not 0 <= start - event["stage"] < iterations:
                continue
            logical_iteration = start - event["stage"]
            _, _, current_copies, current_costs = next(s for s in segments if s[0] <= logical_iteration < s[1])
            for op in event["operations"]:
                if op in by_op:
                    copy = current_copies[op]
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
                warp = _delay(warp, current_costs[op])
        updated = [warp, service]
        for cid in copy_ids:
            updated.append(produced.get(cid, [NEG] * size))
            updated.extend(basis[offsets[cid] + age] for age in range(distance - 1))
        matrix = tuple(tuple(row) for row in updated)
        state = repeat_transition(matrix, end - start, state=state)
    return state[0]
