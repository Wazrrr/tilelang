"""Deterministic Cartesian products with declared axis order."""

from itertools import product


def grid(**axes):
    return [dict(zip(axes, values)) for values in product(*axes.values())]
