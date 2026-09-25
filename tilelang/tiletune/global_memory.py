"""Logical global-memory traffic from actual external tiles and loop visits."""

from math import prod
from tvm.arith import Analyzer, detect_linear_equation
from .src.structural_key import StructuralKey
from .src.ir_utils import _int, _domains, dense_gemms, in_loop, loop_visits
from .src.regions import _bound, _clip_global


def operand_reuse_geometry(col, buffer_facts, loop, depth):
    """Resolve disjoint affine operand tiles in a single dense matrix loop.

    Nominal (unclipped) tile sizes conservatively include boundary padding.
    Other graphs keep the existing per-CTA traffic model.
    """
    if loop is None or _int(loop.extent) is None or depth is None or len(dense_gemms(col)) != 1:
        return None
    axes = [col.block_domains[tag] for tag in sorted(col.block_domains)]
    grid = [_int(domain.extent) for _, domain in axes]
    if not axes or any(size is None or size <= 0 for size in grid):
        return None
    if any(_int(domain.min) != 0 for _, domain in axes):
        return None
    variables = [var for var, _ in axes]
    tiles = []
    for op in col.operations:
        for region in op.reads:
            if region.buffer.scope() != "global":
                continue
            if op.kind != "copy" or op.predicates or not in_loop(op, loop):
                return None
            if any(kind != "4" and not var.same_as(loop.loop_var) for var, _, kind in op.loops):
                return None
            dependencies = set()
            shape = []
            for interval in region.ranges:
                extent = _int(interval.extent)
                coefficients = detect_linear_equation(interval.min, variables)
                if extent is None or extent <= 0 or not coefficients:
                    return None
                strides = [_int(value) for value in coefficients[:-1]]
                if any(value is None for value in strides):
                    return None
                used = [axis for axis, value in enumerate(strides) if value]
                if len(used) > 1 or any(abs(strides[axis]) < extent for axis in used):
                    return None
                dependencies.update(used)
                shape.append(extent)
            dtype = buffer_facts[region.buffer].dtype
            tiles.append(dict(operation=op.index, bytes=(prod(shape) * dtype.bits * dtype.lanes + 7) // 8, axes=sorted(dependencies)))
    if not tiles:
        return None
    swizzle = None
    if col.threadblock_swizzle is not None:
        args = col.threadblock_swizzle.args
        pattern, panel = str(args[0].value), _int(args[1])
        if pattern not in ("rasterization2DRow", "rasterization2DColumn") or not panel or len(grid) < 2:
            return None
        swizzle = dict(pattern=pattern, panel=panel)
    return dict(grid=grid, tiles=tiles, buffer_depth=depth, swizzle=swizzle)


def analyze_global_memory(col, propagated, buffer_facts, *, loop, actual_accesses=False):
    unknown = list(propagated.unknown)

    def tile(region, loops, visits):
        shape = []
        precision = region.precision
        for axis in region.ranges:
            extent = _int(axis.extent)
            if extent is None:
                extent = _int(Analyzer().int_set(axis.extent, _domains(loops)).max_value)
                precision = "conservative"
            shape.append(extent)
        dtype = buffer_facts[region.buffer].dtype
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
        clipped = _clip_global(region, loops)
        entry = tile(clipped, loops, visits["max"])
        entry.update(visits=visits, operation=operation, loop_variables=[str(v) for v, _, kind in loops if kind != "4"])
        if visits["precision"] != "exact":
            entry["precision"] = visits["precision"]
        key = (
            region.buffer,
            tuple((StructuralKey(r.min), StructuralKey(r.extent)) for r in clipped.ranges),
            tuple((v, StructuralKey(r)) for v, r, kind in loops if kind != "4"),
        )
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
    if not outputs:
        unknown.append("no external output tile")
    loop_name = str(loop.loop_var) if loop is not None else None
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
        "unknown": sorted(set(unknown)),
        "precision": "unknown" if unknown else "estimate",
        "assumptions": [
            "logical tile bytes; no transaction/coalescing or inter-block cache model",
            "external accesses are charged per tile visit; internal fused tiles are not global traffic",
        ],
    }
