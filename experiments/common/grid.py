"""Deterministic Cartesian products with declared axis order."""

from itertools import product


def grid(**axes):
    return [dict(zip(axes, values)) for values in product(*axes.values())]


def row_layout_reason(c):
    if "row_threads" in c:
        if c["row_threads"] > c["block_rows"] or c["threads"] % c["row_threads"]:
            return "row thread groups must partition threads and fit block_rows"
        if c["block_cols"] < c["vector"] * (c["threads"] // c["row_threads"]):
            return "column tile cannot supply the declared vector/thread layout"
    return None
