"""Declared configuration domains; imports only the Python standard library."""

from experiments.common.grid import grid as _grid

POLICIES = ("square", "full_row", "full_col")


def current_configurations(workload):
    return _grid(block_M=[64, 128, 32], block_N=[64, 128, 32], num_stages=[0, 2, 3], threads=[128, 256])


def expanded_configurations(w, large):
    yield from _grid(
        implementation=["tiled"],
        block_M=[16, 32, 64, 128],
        block_N=[16, 32, 64, 128, 256],
        num_stages=[0, 1, 2, 3, 4],
        threads=[128, 256],
        qk_policy=POLICIES,
        pv_policy=POLICIES,
        copy_width=[1, 2, 4, 8] if large else [None],
    )


def legacy_configurations():
    """The existing 128-configuration fixed-grid attention experiment."""
    return _grid(block_M=[32, 64, 128, 256], block_N=[32, 64, 128, 256], num_stages=[0, 1, 2, 3], threads=[128, 256])


def protected_configurations(workload):
    """Retain common tiled attention layouts and automatic/vectorized copies."""
    for qk, pv in (("square", "square"), ("full_row", "square"), ("full_row", "full_row")):
        yield from _grid(
            implementation=["tiled"],
            block_M=[32, 64, 128],
            block_N=[32, 64, 128],
            num_stages=[0, 2, 3, 4],
            threads=[128, 256],
            qk_policy=[qk],
            pv_policy=[pv],
            copy_width=[None, 8],
        )


def legality_reason(w, device, c):
    from experiments.common.mma import tile_reason

    return tile_reason(w, device, [(c["block_M"], c["block_N"]), (c["block_M"], w.parameters["dim"])], c["threads"])


def canonical_config(w, c, device=None):
    from experiments.common.mma import mma_partition

    c = dict(c)
    for key, value in dict(qk_policy="full_row", pv_policy="full_row", copy_width=None, implementation="baseline").items():
        c.setdefault(key, value)
    if (
        device is not None
        and device.target.get("arch", "").rstrip("af") == "sm_80"
        and w.dtype in ("float16", "bfloat16")
        and {"block_M", "block_N", "threads"} <= c.keys()
    ):
        # The tiled path explicitly converts the two MMA partitions through shared memory.
        c["implementation"] = (
            "tiled"
            if (
                c["implementation"] == "tiled"
                or c["qk_policy"] != "full_row"
                or c["pv_policy"] != "full_row"
                or c["copy_width"] is not None
            )
            else "baseline"
        )
        c["qk_policy"] = mma_partition(c["block_M"], c["block_N"], c["threads"], c["qk_policy"])
        c["pv_policy"] = mma_partition(c["block_M"], w.parameters["dim"], c["threads"], c["pv_policy"])
    return c
