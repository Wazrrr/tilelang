"""Reusable allocation facts; each stage owns its interpretation of demand."""

from dataclasses import dataclass
from functools import cached_property
from math import prod

from tilelang import tvm

from .ir_utils import _int


@dataclass
class BufferFacts:
    buffer: object
    scope: str
    shape: tuple
    dtype: object
    elements: int | None
    logical_bits: int | None
    layout: object

    @cached_property
    def replication(self):
        return _int(self.layout.replicate_size) if self.layout is not None and hasattr(self.layout, "replicate_size") else 1

    @cached_property
    def owner_threads(self):
        return _int(self.layout.get_thread_size())

    @cached_property
    def local_shape(self):
        return tuple(_int(x) for x in self.layout.get_output_shape())


def collect_buffer_facts(col):
    """Capture shape/dtype once and evaluate layout facts only when needed."""
    facts = {}
    for buffer in col.buffers:
        shape = tuple(_int(x) for x in buffer.shape)
        dtype = tvm.DataType(buffer.dtype)
        elements = prod(shape) if all(x is not None for x in shape) else None
        facts[buffer] = BufferFacts(
            buffer,
            buffer.scope(),
            shape,
            dtype,
            elements,
            elements * dtype.bits * dtype.lanes if elements is not None else None,
            col.layouts.get(buffer.data),
        )
    return facts
