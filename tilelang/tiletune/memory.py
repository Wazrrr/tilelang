"""Capture logical memory work from operation regions and loop dependencies."""

from math import prod

from .src.ir_utils import _int, loop_visits


def analyze_memory_accesses(col, buffer_facts):
    """Count requested accesses before clipping away masks or partial tiles.

    In particular, a 96-wide copy remains 96-wide when its last iteration is
    partial. Bounding its clipped start/end independently can inflate it to the
    whole tensor dimension. Scalar operations keep their enclosing loop visits;
    no assumption of compiler load reuse or memory transactions is made.
    """
    accesses = []
    for op in col.operations:
        external = [(direction, r) for direction in ("reads", "writes") for r in getattr(op, direction) if r.buffer.scope() == "global"]
        if not external:
            continue
        visits = loop_visits(op.loops)
        for direction, region in external:
            extents = [_int(r.extent) for r in region.ranges]
            elements = prod(extents) if all(x is not None and x >= 0 for x in extents) else None
            dtype = buffer_facts[region.buffer].dtype
            accesses.append(
                dict(
                    operation=op.index,
                    direction=direction,
                    buffer=region.buffer.name,
                    buffer_id=str(hash(region.buffer)),
                    bytes=(elements * dtype.bits * dtype.lanes + 7) // 8 if elements is not None else None,
                    visits=visits["max"],
                    visit_precision=visits["precision"],
                    predicated=bool(op.predicates),
                )
            )
    launches = {tuple(sorted((k, str(v)) for k, v in op.launch_threads.items())) for op in col.operations}
    extents = [_int(v) for k, v in col.threads.items() if k.startswith("blockIdx.")]
    grid = prod(extents) if len(launches) == 1 and extents and all(v is not None and v > 0 for v in extents) else None
    return dict(
        accesses=accesses,
        grid_blocks=grid,
        dependencies=[dict(operation=op.index, predecessors=op.dependencies) for op in col.operations],
        unknown=list(col.unknown),
    )
