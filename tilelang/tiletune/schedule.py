"""Buffer readiness/reuse recurrence and launch-wide CTA scheduling."""

from functools import lru_cache
from math import inf, prod
from itertools import groupby
import heapq
from tilelang import tvm
from tvm import tirx as tir
from tvm.arith import Analyzer
from .src.ir_utils import _int, in_loop

NEG = -inf


def _maximum(*rows):
    return [max(values) for values in zip(*rows)]


def _delay(row, cycles):
    return [value + cycles for value in row]


@lru_cache(maxsize=256)
def _square(matrix):
    size = len(matrix)
    return tuple(tuple(max(matrix[i][k] + matrix[k][j] for k in range(size)) for j in range(size)) for i in range(size))


def repeat_transition(matrix, iterations):
    """Max-plus exponentiation; O(log(iterations)) transitions, all slots free at t=0."""
    if not isinstance(iterations, int) or iterations < 0:
        raise ValueError("iteration count must be a nonnegative integer")
    state = [0.0] * len(matrix)
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
        service = _delay(_maximum(issue, service), copy["bytes"] / bandwidth)
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


def collect_producer_buffers(col, loop):
    """Read actual copy regions and consumers; do not infer operand roles by name."""

    inside = [op for op in col.operations if in_loop(op, loop)]
    producers = [
        op
        for op in inside
        if op.kind in ("copy", "async_copy")
        and any(r.buffer.scope() == "global" for r in op.reads)
        and any(r.buffer.scope().startswith("shared") for r in op.writes)
    ]
    copies, targets = [], set()
    for op in producers:
        if len(op.writes) != 1 or op.unknown or op.predicates:
            raise ValueError("unresolved producer destination or conditional copy")
        region = op.writes[0]
        buffer = region.buffer
        if buffer in targets:
            raise ValueError("multiple producer writes to one buffer require a region-aware reuse schedule")
        targets.add(buffer)
        uses = [other.index for other in inside if any(r.buffer.same_as(buffer) for r in other.reads)]
        if (
            not uses
            or min(uses) <= op.index
            or any(other is not op and any(r.buffer.same_as(buffer) for r in other.writes) for other in inside)
        ):
            raise ValueError("unresolved producer/consumer reuse or overwrite")
        dims = [_int(r.extent) for r in region.ranges]
        if any(x is None or x <= 0 for x in dims):
            raise ValueError("symbolic producer tile size")
        dtype = tvm.DataType(buffer.dtype)
        copies.append(
            dict(
                operation=op.index,
                buffer=buffer.name,
                buffer_id=str(hash(buffer)),
                region=region.to_dict(),
                dtype=str(buffer.dtype),
                bytes=(prod(dims) * dtype.bits * dtype.lanes + 7) // 8,
                first_consumer=min(uses),
                last_consumer=max(uses),
            )
        )
    return copies


def collect_cta_work(col, loop, max_axis_points=4096):
    """Compress one varying block axis into runs; never expand the inner loop."""

    domains = col.block_domains
    result = {"precision": "unknown", "groups": [], "repetitions": None, "grid_blocks": None, "unknown": []}
    if (loop is None and (col.pipeline_loops or col.serial_loops)) or not domains:
        result["unknown"].append("no unique launch domain and pipeline loop")
        return result
    axes = sorted(domains)
    sizes = [_int(domains[axis][1].extent) for axis in axes]
    if any(n is None or n <= 0 for n in sizes):
        result["unknown"].append("symbolic CTA launch domain")
        return result
    result["grid_blocks"] = prod(sizes)
    extent = loop.extent if loop is not None else tir.IntImm("int32", 0)
    constant = _int(extent)
    if constant is not None and constant >= 0:
        result.update(
            precision="exact",
            groups=[{"iterations": constant, "count": prod(sizes)}],
            repetitions=1,
            min_iterations=constant,
            max_iterations=constant,
            mean_iterations=float(constant),
        )
        return result
    variables = set()
    tir.stmt_functor.post_order_visit(extent, lambda n: variables.add(n) if isinstance(n, tir.Var) else None)
    varying = [i for i, axis in enumerate(axes) if domains[axis][0] in variables]
    if len(varying) != 1 or variables != {domains[axes[varying[0]]][0]}:
        result["unknown"].append("loop extent must depend on one resolved block axis")
        return result
    axis = varying[0]
    var, domain = domains[axes[axis]]
    start = _int(domain.min)
    if start is None or sizes[axis] > max_axis_points:
        result["unknown"].append("CTA work distribution exceeds bounded axis analysis")
        return result
    analyzer = Analyzer()
    counts = [
        _int(analyzer.simplify(tir.stmt_functor.substitute(extent, {var: tir.IntImm(var.dtype, start + i)}))) for i in range(sizes[axis])
    ]
    if any(n is None or n < 0 for n in counts):
        result["unknown"].append("unresolved per-CTA loop count")
        return result
    inner, outer = prod(sizes[:axis]), prod(sizes[axis + 1 :])
    groups = [{"iterations": n, "count": sum(1 for _ in items) * inner} for n, items in groupby(counts)]
    result.update(
        precision="exact",
        groups=groups,
        repetitions=outer,
        varying_axis=axes[axis],
        min_iterations=min(counts),
        max_iterations=max(counts),
        mean_iterations=sum(counts) / len(counts),
        assumptions=["CTA counts come from the original loop extent; x is the fastest launch axis"],
    )
    return result


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
