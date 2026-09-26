"""Logical memory-work ordering, without compute rates or occupancy prediction."""

PIPELINE_DEPTH_RADIX = 1 << 16
PIPELINE_DEPTH_COUNT = PIPELINE_DEPTH_RADIX - 1
LAUNCH_TARGET_WAVES = 3


def _ceil_div(numerator, denominator):
    return (numerator + denominator - 1) // denominator


def _adjusted_byte_waves(byte_waves, grid_blocks, accesses_per_cta, sm_count):
    """Return the E2-fitted launch-underfill adjustment to byte waves."""
    shortfall = max(0, LAUNCH_TARGET_WAVES * sm_count - grid_blocks)
    denominator = grid_blocks + accesses_per_cta
    return _ceil_div(byte_waves * (denominator + shortfall), denominator)


def _encode_memory_order(adjusted_byte_waves, pipeline_depth, access_waves):
    """Exactly encode the lexicographic key (U, -D, E)."""
    return (
        PIPELINE_DEPTH_COUNT * adjusted_byte_waves * (adjusted_byte_waves + 1) // 2
        + (PIPELINE_DEPTH_COUNT - pipeline_depth) * (adjusted_byte_waves + 1)
        + access_waves
    )


def classify_bound(compute_work, unique_bytes, ridge_flops_per_byte):
    """Classify arithmetic intensity against a positive roofline ridge point.

    This utility remains available for callers that explicitly want a roofline
    label. The pool-normalized ``bound_aware`` ranking does not use it.
    """
    for name, value in (("compute_work", compute_work), ("unique_bytes", unique_bytes)):
        if value is None:
            return None
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer or None")
    if type(ridge_flops_per_byte) not in (int, float) or ridge_flops_per_byte <= 0:
        raise ValueError("ridge_flops_per_byte must be positive")
    if unique_bytes == 0:
        return None
    return "compute" if compute_work / unique_bytes >= ridge_flops_per_byte else "memory"


def score_memory(accesses, grid_blocks, sm_count, pipeline_depth=1, occupancy_penalty=1):
    """Rank by adjusted byte-waves, descending depth, then access-waves.

    ``U`` applies a three-SM-wave launch-underfill penalty to logical byte-waves
    ``B``. Logical accesses per CTA damp the penalty for heavier CTAs. This is
    an E2-fitted ordinal convention, not a service-time or occupancy prediction.
    The exact integer score preserves the lexicographic key ``(U, -D, E)``;
    equal triples stay tied. ``occupancy_penalty`` is a positive integer that
    optionally scales only ``U`` for the bound-aware coarse schedule gate.
    """
    if type(occupancy_penalty) is not int or occupancy_penalty < 1:
        raise ValueError("occupancy_penalty must be a positive integer")
    unknown = []
    for name, value in (("grid_blocks", grid_blocks), ("sm_count", sm_count)):
        if value is None:
            unknown.append(f"unresolved {name}")
        elif type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer or None")
    if type(pipeline_depth) is not int or not 0 < pipeline_depth < PIPELINE_DEPTH_RADIX:
        raise ValueError(f"pipeline_depth must be an integer in [1, {PIPELINE_DEPTH_RADIX - 1}]")
    byte_work = 0
    events = 0
    for access in accesses:
        size, visits = access["bytes"], access["visits"]
        for name, value in (("bytes", size), ("visits", visits)):
            if value is None:
                unknown.append(f"operation {access['operation']}: unresolved {name}")
            elif type(value) is not int or value < 0:
                raise ValueError(f"access {name} must be a nonnegative integer or None")
        if size is not None and visits is not None:
            byte_work += size * visits
            events += visits if size else 0
    waves = _ceil_div(grid_blocks, sm_count) if grid_blocks is not None and sm_count is not None else None
    byte_waves = byte_work * waves if not unknown else None
    event_waves = events * waves if not unknown else None
    shortfall = max(0, LAUNCH_TARGET_WAVES * sm_count - grid_blocks) if grid_blocks is not None and sm_count is not None else None
    adjusted_byte_waves = _adjusted_byte_waves(byte_waves, grid_blocks, events, sm_count) if byte_waves is not None else None
    effective_byte_waves = adjusted_byte_waves * occupancy_penalty if adjusted_byte_waves is not None else None
    # Every nonempty access transfers at least one byte, so 0 <= E <= B <= U.
    # That bound lets triangular U bands encode (U, -D, E) exactly with Python
    # integers, without an arbitrary work bound or float rounding.
    score = _encode_memory_order(effective_byte_waves, pipeline_depth, event_waves) if effective_byte_waves is not None else None
    return {
        "metric": "memory",
        "score": score,
        "tie_break_score": events * waves if not unknown else None,
        "units": "lexicographic memory-order units",
        "formula": (
            "65535 * effective_byte_waves * (effective_byte_waves + 1) // 2 + "
            "(65535 - pipeline_depth) * (effective_byte_waves + 1) + access_waves; "
            "effective_byte_waves = adjusted_byte_waves * occupancy_penalty"
        ),
        "tie_break_formula": "logical_memory_accesses_per_cta * ceil(grid_blocks / sm_count)",
        "logical_global_bytes_per_cta": byte_work if not unknown else None,
        "logical_byte_waves": byte_waves,
        "adjusted_logical_byte_waves": effective_byte_waves,
        "occupancy_penalty": occupancy_penalty,
        "logical_memory_accesses_per_cta": events if not unknown else None,
        "logical_memory_access_waves": event_waves,
        "single_cta_waves": waves,
        "launch_target_waves": LAUNCH_TARGET_WAVES,
        "launch_underfill_shortfall_blocks": shortfall,
        "pipeline_depth": pipeline_depth,
        "precision": "unknown" if unknown else "estimate",
        "unknown": unknown,
        "assumptions": [
            "padded and predicated logical accesses; no transaction, cache, coalescing, or bandwidth model",
            "single-CTA waves account for grid size without predicting physical residency",
            "three SM waves are an E2-fitted ordinal launch target, not a physical occupancy threshold",
            "per-CTA logical access count dampens the launch-underfill penalty applied to byte-waves",
            "deeper buffering orders equal adjusted byte-waves before fewer logical access-waves",
            "exact integer encoding of (adjusted byte-waves, -pipeline depth, access-waves); equal triples share one tail rank",
            "occupancy_penalty scales only the primary byte-wave term; it never removes or rejects a candidate",
        ]
        + (
            ["a coarse resident-warp service penalty ordered the configured schedule; it is not a measured latency"]
            if occupancy_penalty != 1
            else []
        )
        + [
            "storage and scheduling uncertainty do not exclude a resolved memory score",
        ],
    }
