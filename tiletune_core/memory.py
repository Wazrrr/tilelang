"""Logical memory-work ordering, without compute rates or occupancy prediction."""

PIPELINE_DEPTH_RADIX = 1 << 16


def score_memory(accesses, grid_blocks, sm_count, pipeline_depth=1):
    """Rank by byte-waves, pipeline buffering depth, then access count.

    The wave term is an ordering convention that accounts for launch size. It
    does not predict physical residency. Buffering depth only orders candidates
    with identical byte-waves. The secondary event count orders a report within
    an equal composite-score group but never splits that group for pruning.
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
    return {
        "metric": "memory",
        "score": byte_waves * PIPELINE_DEPTH_RADIX + (PIPELINE_DEPTH_RADIX - 1 - pipeline_depth) if byte_waves is not None else None,
        "tie_break_score": events * waves if not unknown else None,
        "units": "lexicographic memory-order units",
        "formula": "logical_byte_waves * 65536 + (65535 - pipeline_depth)",
        "tie_break_formula": "logical_memory_accesses_per_cta * ceil(grid_blocks / sm_count)",
        "logical_global_bytes_per_cta": byte_work if not unknown else None,
        "logical_byte_waves": byte_waves,
        "logical_memory_accesses_per_cta": events if not unknown else None,
        "single_cta_waves": waves,
        "pipeline_depth": pipeline_depth,
        "precision": "unknown" if unknown else "estimate",
        "unknown": unknown,
        "assumptions": [
            "padded and predicated logical accesses; no transaction, cache, coalescing, or bandwidth model",
            "single-CTA waves account for grid size without predicting physical residency",
            "pipeline depth orders equal byte-work candidates without a fitted timing or overlap model",
            "memory events order tied report entries; equal composite scores share one tail rank",
            "storage and scheduling uncertainty do not exclude a resolved memory score",
        ],
    }
