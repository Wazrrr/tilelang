"""Declared configuration domains; imports only the Python standard library."""

from experiments.common.grid import grid as _grid, row_layout_reason

POLICIES = ("square", "full_row", "full_col")


def current_configurations(workload):
    return _grid(block_rows=[1, 2, 4], threads=[128, 256])


def expanded_configurations(w, large):
    if w.op == "elementwise":
        yield from _grid(
            implementation=["tiled"],
            block_rows=[1, 2, 4, 8],
            block_cols=[64, 128, 256, 512, 1024],
            threads=[64, 128, 256, 512],
            vector=[1, 2, 4, 8],
            row_threads=[1, 2, 4] if large else [1, 2],
        )
    else:
        yield from _grid(
            implementation=["streamed"],
            block_rows=[1, 2, 4, 8, 16],
            block_cols=[128, 256, 512, 1024, 2048] if large else [128, 512, 2048],
            threads=[64, 128, 256, 512],
            vector=[1, 2, 4, 8],
            row_threads=[1, 2, 4],
        )


def legality_reason(w, device, c):
    return row_layout_reason(c)


def canonical_config(w, c, device=None):
    return dict(c)
