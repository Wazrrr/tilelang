"""Compiler-free MMA layout constraints shared by matrix kernel families."""


def mma_partition(m, n, threads, policy):
    """sm80 MMA policy resolution, checked against GemmWarpPolicy in tests.

    Keep planning independent of importing TileLang or running layout inference.
    This only resolves the policy's declared warp partition, not register use.
    """
    warps = threads // 32
    if policy == "full_row":
        rows = warps if m % (warps * 16) == 0 else m // 16
        return rows, max(1, warps // rows)
    if policy == "full_col":
        cols = warps if n % (warps * 8) == 0 else n // 8
        return max(1, warps // cols), cols
    pairs = [(r, warps // r) for r in range(1, min(m // 16, warps) + 1) if warps % r == 0 and warps // r <= n // 8]
    return min(pairs, key=lambda p: abs((m / (p[0] * 16)) / (n / (p[1] * 8)) - m / n)) if pairs else (1, 1)


def tile_reason(workload, device, tiles, threads):
    arch = device.target.get("arch", "").rstrip("af")
    if (
        device.target["kind"] == "cuda"
        and arch == "sm_80"
        and workload.dtype in ("float16", "bfloat16")
        and any(m * n < 4 * threads for m, n in tiles)
    ):
        return "output tile cannot accommodate MMA warps"
    return None
