"""Numerical recurrences and compressed launch scheduling; no compiler dependencies."""

from functools import lru_cache
from math import inf
import heapq

NEG = -inf


def _maximum(*rows):
    return [max(values) for values in zip(*rows)]


def _delay(row, cycles):
    return [value + cycles for value in row]


@lru_cache(maxsize=256)
def _square(matrix):
    size = len(matrix)
    return tuple(tuple(max(matrix[i][k] + matrix[k][j] for k in range(size)) for j in range(size)) for i in range(size))


def repeat_transition(matrix, iterations, *, state=None):
    """Max-plus exponentiation; O(log(iterations)) transitions, all slots free at t=0."""
    if not isinstance(iterations, int) or iterations < 0:
        raise ValueError("iteration count must be a nonnegative integer")
    state = [0.0] * len(matrix) if state is None else list(state)
    if len(state) != len(matrix):
        raise ValueError("transition state size must match its matrix")
    while iterations:
        if iterations & 1:
            state = [max(weight + value for weight, value in zip(row, state)) for row in matrix]
        iterations >>= 1
        if iterations:
            matrix = _square(matrix)
    return state


def buffer_transition(copies, consumers, depth, bandwidth, latency, barrier):
    """One iteration's producer issues, byte service, consumer uses and releases.

    The state carries the previous consumer end, producer issue and byte-service
    end, plus a release-time ring for each distinct producer destination buffer.
    A producer can issue asynchronously; byte service is FIFO. Each consumer
    waits only for tiles it actually reads. Register-carried state serializes
    consumer iterations. Shared buffers become reusable after their last read.
    """
    if not copies or not 1 <= depth <= 32:
        raise ValueError("tile-buffer timing requires a known ring depth between 1 and 32")
    size = 3 + len(copies) * depth
    basis = [[0.0 if i == j else NEG for j in range(size)] for i in range(size)]
    consumer, issue, service = basis[:3]
    ready = []
    for index, copy in enumerate(copies):
        issue = _maximum(issue, basis[3 + index * depth])
        service = _delay(_maximum(issue, service), copy.get("service_bytes", copy["bytes"]) / bandwidth)
        ready.append(_delay(service, latency))
    releases = {}
    for op, cycles in consumers:
        required = [ready[i] for i, copy in enumerate(copies) if copy["first_consumer"] == op]
        consumer = _delay(_maximum(consumer, *required), cycles + len(required) * barrier)
        releases[op] = consumer
    updated = [consumer, issue, service]
    for index, copy in enumerate(copies):
        updated.extend(basis[3 + index * depth + offset] for offset in range(1, depth))
        updated.append(releases[copy["last_consumer"]])
    return tuple(tuple(row) for row in updated)


def estimate_grid_cycles(distribution, timing_for_iterations, slots, uniform_waves):
    """Work-conserving dispatch estimate with exact IR work counts.

    For very large nonuniform grids, report a work-plus-tail approximation instead
    of enumerating every CTA. CUDA dispatch order and shared-SM interference are
    assumptions; the work distribution itself is independent of those assumptions.
    """
    if distribution.get("precision") != "exact" or not slots:
        return None
    groups, repetitions = distribution["groups"], distribution["repetitions"]
    timings = {n: timing_for_iterations(n) for n in {g["iterations"] for g in groups}}
    if not timings or any(timing is None for timing in timings.values()):
        return None
    costs = {n: timing["cycles"] for n, timing in timings.items()}
    work = sum(costs[g["iterations"]] * g["count"] for g in groups) * repetitions
    maximum, minimum = max(costs.values()), min(costs.values())
    if len(costs) == 1:
        cycles, method = maximum * uniform_waves, "uniform CTA waves"
    elif distribution["grid_blocks"] <= 262144:
        available = [0.0] * slots
        for _ in range(repetitions):
            for group in groups:
                cost = costs[group["iterations"]]
                for _ in range(group["count"]):
                    heapq.heapreplace(available, available[0] + cost)
        cycles, method = max(available), "CTA dispatch in launch order"
    else:
        cycles, method = max(maximum, work / slots + maximum * (1 - 1 / slots)), "work plus conservative one-CTA tail"
    return dict(
        cycles=cycles,
        method=method,
        work_cycles=work,
        min_cta_cycles=minimum,
        max_cta_cycles=maximum,
        work_lower_bound_cycles=max(maximum, work / slots),
        work_distribution=distribution,
        assumptions=[
            "equal resident slots; effective per-CTA rates share the SM throughput",
            "CUDA dispatch and resource contention are modeled, not observed",
        ],
    )
