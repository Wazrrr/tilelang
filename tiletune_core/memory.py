"""Logical memory-work ordering, without compute rates or occupancy prediction."""

PIPELINE_DEPTH_RADIX = 1 << 16
PIPELINE_DEPTH_COUNT = PIPELINE_DEPTH_RADIX - 1


def score_memory(accesses, grid_blocks, sm_count, pipeline_depth=1):
    """Rank by byte-waves, descending pipeline depth, then access-waves.

    The wave term is an ordering convention that accounts for launch size. It
    does not predict physical residency. Deeper buffering orders equal byte
    work before fewer logical requests. These are ordinal preferences, not
    fitted service-time costs.
    All three components define the primary score; equal triples stay tied.
    """
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
    waves = (grid_blocks + sm_count - 1) // sm_count if grid_blocks is not None and sm_count is not None else None
    byte_waves = byte_work * waves if not unknown else None
    event_waves = events * waves if not unknown else None
    # Every nonempty access transfers at least one byte, so 0 <= events <= bytes.
    # A byte-work band B contains exactly 65535 * (B + 1) possible (depth,
    # event) pairs. Summing all preceding band widths and adding the within-band
    # offset exactly encodes (bytes, -depth, events), without floats or bounds.
    order = (
        PIPELINE_DEPTH_COUNT * byte_waves * (byte_waves + 1) // 2
        + (PIPELINE_DEPTH_COUNT - pipeline_depth) * (byte_waves + 1)
        + event_waves
        if byte_waves is not None
        else None
    )
    return {
        "metric": "memory",
        "score": order,
        "tie_break_score": events * waves if not unknown else None,
        "units": "lexicographic memory-order units",
        "formula": "65535 * byte_waves * (byte_waves + 1) // 2 + (65535 - pipeline_depth) * (byte_waves + 1) + access_waves",
        "tie_break_formula": "logical_memory_accesses_per_cta * ceil(grid_blocks / sm_count)",
        "logical_global_bytes_per_cta": byte_work if not unknown else None,
        "logical_byte_waves": byte_waves,
        "logical_memory_accesses_per_cta": events if not unknown else None,
        "logical_memory_access_waves": event_waves,
        "single_cta_waves": waves,
        "pipeline_depth": pipeline_depth,
        "precision": "unknown" if unknown else "estimate",
        "unknown": unknown,
        "assumptions": [
            "padded and predicated logical accesses; no transaction, cache, coalescing, or bandwidth model",
            "single-CTA waves account for grid size without predicting physical residency",
            "deeper buffering orders equal byte-work candidates before fewer logical requests",
            "exact integer encoding of (byte-waves, -pipeline depth, access-waves); equal triples share one tail rank",
            "storage and scheduling uncertainty do not exclude a resolved memory score",
        ],
    }
