"""CTA resource limits and launch waves, independent of kernel family."""

from math import prod


def analyze_waves(col, memory, pressure, device_limits=None):
    from .src.ir_utils import _int

    unknown = []
    model = pressure.get("target_model")
    if model is not None and model["kind"] is not None and not model["block_execution"]:
        unknown.append("target requires a non-SIMT core/storage residency model")
    limits = dict(device_limits or {})
    smem = memory["shared_memory_bytes_estimate"]
    shared = memory["shared_allocations"]

    def product(values):
        values = [_int(v) for v in values]
        return prod(values) if all(v is not None and v >= 0 for v in values) else None

    thread_sets = {tuple(sorted((k, str(v)) for k, v in op.launch_threads.items())) for op in col.operations}
    grid = product(v for k, v in col.threads.items() if k.startswith("blockIdx."))
    threads = product(v for k, v in col.threads.items() if k.startswith("threadIdx."))
    if len(thread_sets) != 1 or not any(k.startswith("threadIdx.") for k in col.threads):
        grid = threads = None
        unknown.append("unresolved or multiple launch domains")
    if not shared and smem == 0:
        smem = 0
    original_threads = threads
    ws = pressure.get("warp_specialization", {})
    if ws.get("status") == "predicted":
        threads = ws["launch_threads"]
    elif ws.get("status") == "unknown":
        unknown.append("unresolved Hopper warp-specialization policy")
    bounds = {}
    if threads and limits.get("max_threads_per_sm"):
        warp = limits.get("warp_size", 1)
        rounded_threads = ((threads + warp - 1) // warp) * warp
        bounds["threads"] = limits["max_threads_per_sm"] // rounded_threads
    if smem is not None and limits.get("shared_memory_per_sm"):
        bounds["shared_memory"] = limits["shared_memory_per_sm"] // max(smem, 1)
    block_regs = (
        max(
            pressure.get("modeled_accumulator_registers_per_block") or 0,
            (pressure.get("tile_liveness") or {}).get("peak_registers_per_block_estimate") or 0,
        )
        or None
    )
    tile_regs = block_regs
    register_basis = "tile-state proxy; physical allocation is unknown"
    physical = pressure.get("physical_register_allocation", {})
    if ws.get("status") == "predicted":
        block_regs = physical.get("registers_per_block")
        register_basis = "producer/consumer policy reservation; independent of logical tile demand"
    if block_regs and threads and limits.get("registers_per_sm"):
        bounds["registers"] = limits["registers_per_sm"] // block_regs
    if limits.get("max_blocks_per_sm"):
        bounds["blocks"] = limits["max_blocks_per_sm"]
    required = {"sm_count", "shared_memory_per_sm", "registers_per_sm", "max_threads_per_sm", "max_blocks_per_sm", "warp_size"}
    if not required <= limits.keys() or not block_regs or smem is None or not threads:
        unknown.append("incomplete occupancy inputs")
    if (limits.get("shared_memory_per_block") and smem is not None and smem > limits["shared_memory_per_block"]) or (
        limits.get("max_threads_per_block") and threads and threads > limits["max_threads_per_block"]
    ):
        unknown.append("estimated block resources exceed device limits")
    resident = min(bounds.values()) if bounds else None
    waves = (
        (grid + resident * limits["sm_count"] - 1) // (resident * limits["sm_count"])
        if resident and grid and limits.get("sm_count")
        else None
    )
    if not waves:
        unknown.append("unresolved wave count")
    return {
        "grid_blocks": grid,
        "launch_threads": threads,
        "original_launch_threads": original_threads,
        "registers_per_block_estimate": block_regs,
        "logical_tile_registers_per_block_estimate": tile_regs,
        "register_estimate_basis": register_basis,
        "device_limits": limits,
        "resident_blocks_limits": bounds,
        "resident_blocks_per_sm_estimate": resident,
        "num_waves_estimate": waves,
        "unknown": sorted(set(unknown)),
        "precision": "unknown" if unknown else "estimate",
        "assumptions": [
            "occupancy is an estimate; compiler initial allocation, scratch and granularity are unmodeled",
            "known WS policies use physical reservations; otherwise tile storage is only an occupancy proxy",
            "the register demand allowance never changes physical register residency",
        ]
        + (
            [
                "HIP uses aggregate logical vector/accumulator storage as a proxy; scalar-register limits and per-SIMD allocation granularity are unmodeled"
            ]
            if model and model["kind"] == "hip"
            else []
        ),
    }
