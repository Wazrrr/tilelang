"""Declared configuration domains; imports only the Python standard library."""

from experiments.common.grid import grid as _grid

POLICIES = ("square", "full_row", "full_col")


def current_configurations(workload):
    if workload.op == "kda_recurrent":
        return _grid(block_v=[16, 32, 64], threads=[128, 256]) + _grid(
            implementation=["tiled"], block_v=[16, 32, 64], threads=[128, 256], block_t=[4, 16, 32], stages=[0, 2, 3], unroll=[1, 4]
        )
    else:
        tiled = _grid(
            implementation=["tiled"],
            block_m=[16, 32, 64],
            block_k=[32, 64],
            block_v=[32, 64],
            block_s=[16, 32, 64],
            stages=[0, 2, 3],
            intra_stages=[0, 2],
            threads=[128, 256],
        )
        return _grid(block_k=[32, 64], block_v=[32, 64], stages=[0, 2, 3], threads=[128, 256]) + [
            c for c in tiled if c["block_m"] * c["block_v"] >= 4 * c["threads"]
        ]


def expanded_configurations(w, large):
    if w.op == "kda_recurrent":
        yield from _grid(
            implementation=["tiled"],
            block_v=[8, 16, 32, 64, 128],
            block_t=[1, 2, 4, 8, 16, 32, 64],
            stages=[0, 1, 2, 3, 4] if large else [0, 1, 2, 3],
            unroll=[1, 2, 4, 8],
            threads=[64, 128, 256],
        )
    else:
        yield from _grid(
            implementation=["tiled"],
            block_m=[16, 32, 64, 128],
            block_k=[16, 32, 64, 128] if large else [16, 32, 64],
            block_v=[32, 64, 128],
            block_s=[16, 32, 64, 128],
            stages=[0, 1, 2, 3, 4] if large else [0, 1, 2, 3],
            intra_stages=[0, 1, 2, 3] if large else [0, 1, 2],
            threads=[128, 256],
        )


def legality_reason(w, device, c):
    from experiments.common.mma import tile_reason

    if w.op == "kda_recurrent":
        if c.get("implementation") == "tiled" and c["block_t"] % c["unroll"]:
            return "unroll must divide block_t"
        return None
    return tile_reason(w, device, [(c.get("block_m", w.parameters["chunk_size"]), c["block_v"])], c["threads"])


def canonical_config(w, c, device=None):
    c = dict(c)
    defaults = (
        dict(implementation="baseline", block_t=4, stages=0, unroll=1)
        if w.op == "kda_recurrent"
        else dict(implementation="baseline", block_m=32, block_s=32, intra_stages=0)
    )
    for key, value in defaults.items():
        c.setdefault(key, value)
    return c
