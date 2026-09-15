"""Declared configuration domains; imports only the Python standard library."""

from experiments.common.grid import grid as _grid

POLICIES = ("square", "full_row", "full_col")


def current_configurations(workload):
    return _grid(block_m=[32, 64, 128], block_n=[32, 64, 128], block_k=[32, 64], stages=[0, 2, 3], threads=[128, 256])


def expanded_configurations(w, large):
    yield from _grid(
        block_m=[16, 32, 64, 128, 256],
        block_n=[16, 32, 64, 128, 256],
        block_k=[16, 32, 64, 128],
        stages=[0, 1, 2, 3, 4],
        threads=[128, 256],
        warp_policy=POLICIES,
        swizzle_panel=[0, 4, 8] if large else [0],
    )


def advanced_configurations():
    """The existing 288-configuration advanced-autotune implementation."""
    return _grid(
        block_M=[64, 128, 256],
        block_N=[64, 128, 256],
        block_K=[32, 64],
        num_stages=[0, 1, 2, 3],
        thread_num=[128, 256],
        enable_rasteration=[True, False],
    )


def legality_reason(w, device, c):
    from experiments.common.mma import tile_reason

    return tile_reason(w, device, [(c["block_m"], c["block_n"])], c["threads"])


def canonical_config(w, c, device=None):
    from experiments.common.mma import mma_partition

    c = dict(c)
    c.setdefault("warp_policy", "square")
    c.setdefault("swizzle_panel", 0)
    if (
        device is not None
        and device.target.get("arch", "").rstrip("af") == "sm_80"
        and w.dtype in ("float16", "bfloat16")
        and {"block_m", "block_n", "threads"} <= c.keys()
    ):
        c["warp_policy"] = mma_partition(c["block_m"], c["block_n"], c["threads"], c["warp_policy"])
    return c
