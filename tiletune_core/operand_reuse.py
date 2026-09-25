"""L2 service for affine operands shared by a resident CTA cohort."""

from math import prod


def operand_read_service(geometry, profile, cohort):
    """Bound traffic using the worst contiguous cohort in CUDA launch order.

    CTAs are assumed to advance through the same matrix loop together. Only
    current buffered operand tiles are retained; no reuse across waves is
    credited. The L2 and DRAM throughput limits apply simultaneously.
    """
    if not geometry or not profile.get("l2_bytes_per_cycle") or not profile.get("l2_cache_bytes"):
        return None
    grid, tiles = geometry["grid"], geometry["tiles"]
    count = prod(grid)
    cohort = min(count, cohort)
    swizzle = geometry["swizzle"]
    peak = 0
    for start in sorted({*range(0, count - cohort + 1, cohort), count - cohort}):
        coordinates = []
        for linear in range(start, start + cohort):
            remainder, point = linear, []
            for extent in grid:
                point.append(remainder % extent)
                remainder //= extent
            if swizzle:
                # Same integer mapping as cuda/threadblock_swizzle.h.
                x, y = point[:2]
                width = swizzle["panel"]
                column = swizzle["pattern"] == "rasterization2DColumn"
                major, minor = (grid[1], grid[0]) if column else (grid[0], grid[1])
                panel, offset = divmod(x + y * grid[0], width * major)
                stride = min(width, minor - panel * width)
                slow, fast = offset // stride, offset % stride + panel * width
                if panel % 2:
                    slow = major - 1 - slow
                point[:2] = (fast, slow) if column else (slow, fast)
            coordinates.append(point)
        unique = sum(tile["bytes"] * len({tuple(point[axis] for axis in tile["axes"]) for point in coordinates}) for tile in tiles)
        peak = max(peak, unique)
    logical = cohort * sum(tile["bytes"] for tile in tiles)
    working_set = peak * geometry["buffer_depth"]
    fits = working_set <= profile["l2_cache_bytes"]
    fraction = peak / logical if fits else 1.0
    dram = profile["global_bytes_per_cycle"]
    rate = max(dram, min(profile["l2_bytes_per_cycle"], dram / fraction))
    return dict(
        read_bytes_per_cycle=rate,
        dram_fraction=fraction,
        cohort_ctas=cohort,
        logical_bytes_per_iteration=logical,
        unique_bytes_per_iteration=peak,
        buffered_working_set_bytes=working_set,
        fits_l2=fits,
        assumptions=["resident CTAs traverse the same loop together", "no operand reuse across CTA waves", "writes retain DRAM service"],
    )
