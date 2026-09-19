"""Logical memory-work ordering, without compute rates or occupancy prediction."""


def score_memory(accesses, grid_blocks, sm_count):
    """Rank by bytes per single-CTA wave, then logical memory-access count.

    Each access describes a tile transfer or a scalar access and its loop visits.
    Predicates and tail masks may suppress work: these are logical upper estimates,
    not measured DRAM traffic. One CTA per SM is an ordering convention, not an
    occupancy claim. The secondary key orders equal-score report entries by
    transfer count; it does not split their conservative rank group. No hardware
    rates are needed.
    """
    unknown = []
    for name, value in (("grid_blocks", grid_blocks), ("sm_count", sm_count)):
        if value is None:
            unknown.append(f"unresolved {name}")
        elif type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer or None")
    byte_work, events = 0, 0
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
    return dict(
        metric="memory",
        score=byte_work * waves if not unknown else None,
        tie_break_score=events * waves if not unknown else None,
        units="logical byte-waves",
        formula="logical_global_bytes_per_cta * ceil(grid_blocks / sm_count)",
        tie_break_formula="logical_memory_accesses_per_cta * ceil(grid_blocks / sm_count)",
        logical_global_bytes_per_cta=byte_work if not unknown else None,
        logical_memory_accesses_per_cta=events if not unknown else None,
        single_cta_waves=waves,
        precision="unknown" if unknown else "estimate",
        unknown=unknown,
        assumptions=[
            "padded and predicated logical accesses; no transaction, cache, coalescing or bandwidth model",
            "single-CTA waves account for grid size without predicting physical residency",
            "memory events order tied report entries; primary-score ties share one tail rank; no compute or overlap timing",
            "storage and scheduling uncertainty do not exclude a resolved memory score",
        ],
    )
