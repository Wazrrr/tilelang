"""Logical memory-work ordering, without compute rates or occupancy prediction."""

PIPELINE_DEPTH_RADIX = 1 << 16
PIPELINE_DEPTH_COUNT = PIPELINE_DEPTH_RADIX - 1
LAUNCH_TARGET_WAVES = 3


def _ceil_div(numerator, denominator):
    return (numerator + denominator - 1) // denominator


def _adjusted_byte_waves(byte_waves, grid_blocks, accesses_per_cta, sm_count):
    """Return the three-SM-wave launch-underfill adjustment to byte waves.

    ``byte_waves`` already multiplies logical per-CTA bytes by the launch wave
    count. Grids smaller than three full SM waves are penalized; heavier CTAs
    (more logical accesses per CTA) damp that penalty. This is the h200-new
    ``U`` term.
    """
    shortfall = max(0, LAUNCH_TARGET_WAVES * sm_count - grid_blocks)
    denominator = grid_blocks + accesses_per_cta
    return _ceil_div(byte_waves * (denominator + shortfall), denominator)


def _encode_memory_order(adjusted_byte_waves, waves, pipeline_depth, access_waves, include_launch_waves=True):
    """Exactly encode a lexicographic memory-order key.

    ``include_launch_waves=True`` encodes the four-level key
    ``(U, waves, -D, E)``. ``False`` encodes the three-level key ``(U, -D, E)``:
    the launch-wave count is dropped because ``U`` already multiplies by it, so
    the bound-aware ordering does not charge the same wave count twice.

    For every nonempty access ledger ``0 <= E <= B <= U``, and when bytes are
    nonempty ``waves <= U``. A variable base of ``U + 1`` therefore covers the
    access component without an arbitrary upper bound, while pipeline depth
    keeps the fixed radix.
    """
    base = adjusted_byte_waves + 1
    depth_inv = PIPELINE_DEPTH_COUNT - pipeline_depth
    if include_launch_waves:
        return ((adjusted_byte_waves * base + waves) * PIPELINE_DEPTH_RADIX + depth_inv) * base + access_waves
    return (adjusted_byte_waves * PIPELINE_DEPTH_RADIX + depth_inv) * base + access_waves


def classify_bound(compute_work, unique_bytes, ridge_flops_per_byte):
    """Split a kernel into ``compute`` or ``memory`` bound from arithmetic intensity.

    ``compute_work`` is a nonnegative FLOP count and ``unique_bytes`` the
    distinct input/output bytes. The ratio is compared with the device ridge
    point ``theoretical_peak_flops / theoretical_peak_bandwidth``. This is a
    coarse roofline split for choosing which lightweight analysis to run; it is
    not a calibrated service-time model, and an unresolved input yields
    ``None`` so the caller can keep the conservative memory ordering.
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
    intensity = compute_work / unique_bytes
    return "compute" if intensity >= ridge_flops_per_byte else "memory"


def score_memory(accesses, grid_blocks, sm_count, pipeline_depth=1, occupancy_penalty=1, include_launch_waves=True):
    """Rank by adjusted byte-waves, then depth and access-waves.

    The primary component ``U`` is the h200-new adjusted byte-wave term. With
    ``include_launch_waves=True`` the next component is the launch-wave count,
    followed by deeper buffering and then fewer logical memory requests; with
    ``include_launch_waves=False`` the launch-wave component is dropped and the
    key is ``(U, -D, E)``. These are ordinal preferences, not fitted
    service-time costs.

    ``occupancy_penalty`` optionally multiplies the primary byte-wave term. It
    is a coarse, piecewise-constant service penalty for configurations whose
    resident warps cannot hide the latency of the configured schedule. It is a
    positive integer so the ordinal comparison stays exact; callers that only
    model memory ordering leave it at its neutral value ``1``.
    """
    if type(occupancy_penalty) is not int or occupancy_penalty < 1:
        raise ValueError("occupancy_penalty must be a positive integer")
    if not isinstance(include_launch_waves, bool):
        raise ValueError("include_launch_waves must be a bool")
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
    byte_waves = byte_work * waves if waves is not None and not unknown else None
    event_waves = events * waves if waves is not None and not unknown else None
    shortfall = (
        max(0, LAUNCH_TARGET_WAVES * sm_count - grid_blocks)
        if grid_blocks is not None and sm_count is not None
        else None
    )
    adjusted_byte_waves = (
        _adjusted_byte_waves(byte_waves, grid_blocks, events, sm_count)
        if byte_waves is not None and grid_blocks is not None and sm_count is not None
        else None
    )
    effective_byte_waves = adjusted_byte_waves * occupancy_penalty if adjusted_byte_waves is not None else None
    # A ledger with no bytes has U == 0, so waves cannot fit the U + 1 base.
    # Such kernels have no global-memory work to order by wave count.
    encoded_waves = waves if byte_work > 0 else 0
    score = (
        _encode_memory_order(effective_byte_waves, encoded_waves, pipeline_depth, event_waves, include_launch_waves)
        if effective_byte_waves is not None and event_waves is not None
        else None
    )

    return {
        "metric": "memory",
        "score": score,
        "tie_break_score": event_waves,
        "units": "lexicographic memory-order units",
        "formula": (
            "((effective_byte_waves * (effective_byte_waves + 1) + waves) * 65536 + "
            "(65535 - pipeline_depth)) * (effective_byte_waves + 1) + access_waves; "
            "effective_byte_waves = adjusted_byte_waves * occupancy_penalty"
            if include_launch_waves
            else "(effective_byte_waves * 65536 + (65535 - pipeline_depth)) * "
            "(effective_byte_waves + 1) + access_waves; "
            "effective_byte_waves = adjusted_byte_waves * occupancy_penalty"
        ),
        "tie_break_formula": "logical_memory_accesses_per_cta * ceil(grid_blocks / sm_count)",
        "logical_global_bytes_per_cta": byte_work if not unknown else None,
        "logical_byte_waves": byte_waves,
        "adjusted_logical_byte_waves": effective_byte_waves,
        "occupancy_penalty": occupancy_penalty,
        "launch_waves_component": include_launch_waves,
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
            "fewer launch waves order equal adjusted byte-waves; deeper buffering orders equal adjusted byte-waves and waves",
            "fewer access-waves order otherwise equal candidates",
            "exact integer encoding of (adjusted byte-waves, waves, -pipeline depth, access-waves) or (adjusted byte-waves, -pipeline depth, access-waves); equal tuples share one tail rank",
            "occupancy_penalty scales only the primary byte-wave term; it never removes or rejects a candidate",
        ]
        + (
            []
            if occupancy_penalty == 1
            else [
                "a coarse resident-warp service penalty ordered the configured schedule; it is not a measured latency",
            ]
        )
        + [
            "storage and scheduling uncertainty do not exclude a resolved memory score",
        ],
    }
