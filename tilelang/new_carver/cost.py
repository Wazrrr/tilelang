"""Tile-level traffic, shared storage and optimistic occupancy for ranking.

These estimates never reject a configuration. No compiler counters or measured
latencies enter the score. Each external tile is charged once per loop visit;
inter-CTA cache reuse, transactions, barriers and compiler temporaries are not
predicted. Unknown inputs make the score unknown, not zero.
"""

DEVICE_LIMIT_FIELDS = {
    "sm_count",
    "shared_memory_per_sm",
    "shared_memory_per_block",
    "registers_per_sm",
    "max_threads_per_sm",
    "max_blocks_per_sm",
    "max_threads_per_block",
    "warp_size",
}


def query_device_limits(target=None):
    """Read the current CUDA device only when it matches the compilation target."""
    import torch
    from .budget import resolve_register_budget
    from .config import CarverConfig

    if not torch.cuda.is_available():
        return None
    arch = resolve_register_budget(CarverConfig(), target)["target_arch"]
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    if arch is None or arch.removeprefix("sm_").rstrip("af") != f"{props.major}{props.minor}":
        return None
    from tilelang.carver.arch.driver.cuda_driver import get_max_blocks_per_multiprocessor

    values = {
        "sm_count": props.multi_processor_count,
        "shared_memory_per_sm": props.shared_memory_per_multiprocessor,
        "shared_memory_per_block": props.shared_memory_per_block_optin,
        "registers_per_sm": props.regs_per_multiprocessor,
        "max_threads_per_sm": props.max_threads_per_multi_processor,
        "max_threads_per_block": props.max_threads_per_block,
        "warp_size": props.warp_size,
        "max_blocks_per_sm": get_max_blocks_per_multiprocessor(device),
    }
    return {k: int(v) for k, v in values.items() if v is not None and v > 0}


def analyze_tile_cost(col, propagated, pressure, device_limits=None, context=None, specialization=None):
    from .memory import analyze_memory
    from .waves import analyze_waves

    memory = specialization.memory_traffic(context) if specialization is not None else analyze_memory(col, propagated)
    waves = analyze_waves(col, memory, pressure, device_limits)
    return {**combine_tile_cost(memory, waves), "_memory": memory, "_waves": waves}


def combine_tile_cost(memory, waves):
    traffic, grid = memory["traffic_bytes_per_block"], waves["grid_blocks"]
    unknown = sorted(set(memory["unknown"] + waves["unknown"]))
    count = waves["num_waves_estimate"]
    score = (traffic + 1) * count if not unknown and traffic is not None and count is not None else None
    return {
        **memory,
        **waves,
        "traffic_bytes_grid_estimate": traffic * grid if traffic is not None and grid is not None else None,
        "score": score,
        "score_formula": "(traffic_bytes_per_block + 1) * num_waves_estimate",
        "precision": "unknown" if unknown else "estimate",
        "unknown": unknown,
        "assumptions": memory["assumptions"] + waves["assumptions"] + ["ranking never adds rejections or truncates configs"],
    }


# Compatibility re-export for existing new-Carver callers.
from .ranking import rank_records  # noqa: F401, E402
