"""Periodic tile-buffer scheduling without expanding loop iterations."""

from functools import lru_cache
from math import inf


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
    from math import prod
    from tilelang import tvm
    from .analysis import _int
    from .specializations import in_loop

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
