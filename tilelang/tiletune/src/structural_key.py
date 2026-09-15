"""Collision-checked IR keys without diagnostic printing."""

from tilelang import tvm
from collections.abc import Mapping


class StructuralKey:
    def __init__(self, value):
        self.value = value
        self.code = tvm.ir.structural_hash(value)

    def __hash__(self):
        return self.code

    def __eq__(self, other):
        return isinstance(other, StructuralKey) and self.code == other.code and tvm.ir.structural_equal(self.value, other.value)


def context_key(value):
    """Snapshot configuration values without conflating types or printed IR."""
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return type(value), value
    if isinstance(value, Mapping):
        return "mapping", frozenset((context_key(k), context_key(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return type(value), tuple(context_key(v) for v in value)
    return StructuralKey(value)
