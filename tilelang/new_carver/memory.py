"""Tile input propagation, actual memory accesses and shared storage."""

from math import prod
from tilelang import tvm
from tvm.arith import Analyzer


def loop_visits(loops):
    from .analysis import _int, _domains

    expression = prod(r.extent for _, r, kind in loops if kind != "4")
    value = _int(expression)
    if value is not None:
        return {"min": value, "max": value, "expression": str(expression), "precision": "exact"}
    bounds = Analyzer().int_set(expression, _domains(loops))
    lower, upper = _int(bounds.min_value), _int(bounds.max_value)
    return {
        "min": lower,
        "max": upper,
        "expression": str(expression),
        "precision": "conservative" if lower is not None and upper is not None else "unknown",
    }


def analyze_memory(col, propagated, specialization=None, actual_accesses=False, pass_configs=None):
    from .analysis import _int, _bound, _domains, _clip_global

    unknown = list(propagated.unknown)

    def product(values):
        values = [_int(v) for v in values]
        return prod(values) if all(v is not None and v >= 0 for v in values) else None

    def tile(region, loops, visits):
        shape = []
        precision = region.precision
        for axis in region.ranges:
            extent = _int(axis.extent)
            if extent is None:
                extent = _int(Analyzer().int_set(axis.extent, _domains(loops)).max_value)
                precision = "conservative"
            shape.append(extent)
        dtype = tvm.DataType(region.buffer.dtype)
        elements = prod(shape) if all(v is not None and v >= 0 for v in shape) else None
        nbytes = (elements * dtype.bits * dtype.lanes + 7) // 8 if elements is not None else None
        if nbytes is None or visits is None:
            unknown.append(f"unresolved tile volume/visits: {region.buffer.name}")
        return {
            **region.to_dict(),
            "precision": precision,
            "dtype": str(region.buffer.dtype),
            "tile_bytes": nbytes,
            "visits_per_block": visits,
            "bytes_per_block": nbytes * visits if nbytes is not None and visits is not None else None,
        }

    inputs, outputs = [], []
    seen = set()
    sources = [(region, loops, None) for region, loops in zip(propagated.per_iteration_inputs, propagated.input_loops)]
    if actual_accesses:
        sources = [(region, op.loops, op.index) for op in col.operations for region in op.reads if region.buffer.scope() == "global"]
    for region, loops, operation in sources:
        visits = loop_visits(loops)
        entry = tile(_clip_global(region, loops), loops, visits["max"])
        entry.update(visits=visits, operation=operation, loop_variables=[str(v) for v, _, kind in loops if kind != "4"])
        if visits["precision"] != "exact":
            entry["precision"] = visits["precision"]
        key = (hash(region.buffer), str(entry["ranges"]), str([(str(v), str(r)) for v, r, kind in loops if kind != "4"]))
        if actual_accesses:
            key = (*key, operation)
        if key not in seen:
            inputs.append(entry)
            seen.add(key)
    for op in col.operations:
        for region in op.writes:
            if region.buffer.scope() != "global":
                continue
            loops = tuple(loop for loop in op.loops if loop[2] != "4")
            # Scalar operations describe a tile's axes, not one memory
            # transaction per thread. Broadcast inputs stay smaller tiles.
            if op.kind == "elementwise":
                region = _bound(region, _domains(loops))
                visits = 1
            else:
                visits = loop_visits(op.loops)["max"]
            outputs.append(tile(_clip_global(region, op.loops), op.loops, visits))

    def byte_sum(entries, key):
        return sum(e[key] for e in entries) if all(e[key] is not None for e in entries) else None

    read_bytes = byte_sum(inputs, "bytes_per_block")
    write_bytes = byte_sum(outputs, "bytes_per_block")
    traffic = read_bytes + write_bytes if read_bytes is not None and write_bytes is not None else None
    shared = []
    for buffer in col.buffers:
        if not buffer.scope().startswith("shared") or buffer.scope() == "shared.tmem":
            continue
        dtype = tvm.DataType(buffer.dtype)
        elements = product(buffer.shape)
        logical = (elements * dtype.bits * dtype.lanes + 7) // 8 if elements is not None else None
        factors = [1]
        for op in col.operations:
            if any(r.buffer.same_as(buffer) for r in op.writes):
                factors.append(product(max(1, s) if s is not None else None for s in op.pipeline_stages))
        stages = max(factors) if all(x is not None for x in factors) else None
        shared.append(
            {
                "buffer": buffer.name,
                "buffer_id": str(hash(buffer)),
                "dtype": str(buffer.dtype),
                "shape": [str(x) for x in buffer.shape],
                "logical_bytes": logical,
                "pipeline_copies_estimate": stages,
                "allocated_bytes_estimate": logical * stages if logical is not None and stages is not None else None,
            }
        )
    from .shared_storage import shared_storage_plan

    storage = shared_storage_plan(col, shared, pass_configs)
    smem = storage["arena_bytes_estimate"]
    if not outputs:
        unknown.append("no external output tile")
    loop_name = str(specialization.loop.loop_var) if specialization is not None and specialization.loop is not None else None
    repeating = [tile for tile in inputs if loop_name is None or loop_name in tile["loop_variables"]]
    once = [tile for tile in inputs if loop_name is not None and loop_name not in tile["loop_variables"]]
    return {
        "input_tiles": inputs,
        "output_tiles": outputs,
        "per_iteration_input_bytes": byte_sum(repeating, "tile_bytes"),
        "one_time_input_bytes": byte_sum(once, "bytes_per_block"),
        "input_working_set_bytes": byte_sum(inputs, "tile_bytes"),
        "cta_work_uniform": all(tile["visits"]["min"] == tile["visits"]["max"] for tile in inputs),
        "input_bytes_per_block": read_bytes,
        "output_bytes_per_block": write_bytes,
        "traffic_bytes_per_block": traffic,
        "shared_allocations": shared,
        "shared_memory_bytes_estimate": smem,
        "shared_memory_allocated_sum_bytes": storage["allocated_sum_bytes"],
        "shared_storage_plan": storage,
        "unknown": sorted(set(unknown)),
        "precision": "unknown" if unknown else "estimate",
        "assumptions": [
            "logical tile bytes; no transaction/coalescing or inter-block cache model",
            "external accesses are charged per tile visit; internal fused tiles are not global traffic",
            "only shared buffers written in a pipeline receive stage copies; layout padding and compiler barriers are unmodeled",
            "shared tile lifetimes predict storage reuse across disjoint regions; entire repeated loops remain overlapping",
            "shared arena size is an estimate, not a compiler allocation guarantee or a rejection bound",
        ],
    }
