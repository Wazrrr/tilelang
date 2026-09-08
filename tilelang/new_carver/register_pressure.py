"""Coordinate the initial register-pressure report from captured IR demands.

Allocation estimates, the dense MMA proof, liveness and rejection policy live in
separate modules. The engine later adds family loop liveness and the predicted
warp-specialization partition before resolving the final register decision.
"""

from .liveness import analyze_allocation_live_sets
from .register_accumulator import analyze_accumulator_bound
from .register_storage import analyze_register_storage

# Compatibility import for callers of the original combined module.
from .liveness import analyze_live_tiles as analyze_live_tiles


def analyze_register_pressure(col):
    """Describe register demand; the engine resolves capacity and rejection."""
    storage = analyze_register_storage(col)
    accumulator = analyze_accumulator_bound(col, storage.modeled_buffers)
    live_sets = analyze_allocation_live_sets(col, storage.modeled_buffers)

    return {
        "logical_storage": storage.logical_storage,
        "live_tile_sets": live_sets,
        "modeled_lower_bound": accumulator.registers_per_thread or None,
        "modeled_accumulator_registers_per_block": accumulator.registers_per_block or None,
        "total_register_upper_bound": None,
        "evidence": accumulator.evidence,
        "assumptions": [
            "compiler operand fragments and temporaries are unmodeled",
            "automatic fragment layout is unresolved; recognized warp-specialization policy is reported separately",
            "tile-state bounds are not bounds on total compiler registers",
        ],
    }
