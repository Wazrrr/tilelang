"""Carver template for KDA chunk-output's two fused matrix products."""

from dataclasses import dataclass

from tvm import te

from .base import BaseTemplate
from ..utils import get_roller_hints_from_func


@dataclass
class KDAChunkTemplate(BaseTemplate):
    batch_size: int = 1
    num_heads: int = 1
    sequence: int = 1
    key_dim: int = 1
    value_dim: int = 1
    chunk_size: int = 1
    in_dtype: str = "float16"
    out_dtype: str = "float16"
    accum_dtype: str = "float32"

    def initialize_function(self) -> None:
        values = (self.batch_size, self.num_heads, self.sequence, self.key_dim, self.value_dim, self.chunk_size)
        if any(type(value) is not int or value <= 0 for value in values) or self.sequence % self.chunk_size:
            raise ValueError("KDA dimensions must be positive and sequence must contain complete chunks")
        groups = self.batch_size * self.num_heads * (self.sequence // self.chunk_size)
        q = te.placeholder((groups, self.chunk_size, self.key_dim), name="Q", dtype=self.in_dtype)
        hidden = te.placeholder((groups, self.key_dim, self.value_dim), name="H", dtype=self.in_dtype)
        a = te.placeholder((groups, self.chunk_size, self.chunk_size), name="A", dtype=self.in_dtype)
        v = te.placeholder((groups, self.chunk_size, self.value_dim), name="V", dtype=self.in_dtype)
        rk = te.reduce_axis((0, self.key_dim), name="rk")
        rs = te.reduce_axis((0, self.chunk_size), name="rs")
        state = te.compute(
            (groups, self.chunk_size, self.value_dim),
            lambda g, i, j: te.sum(q[g, i, rk].astype(self.accum_dtype) * hidden[g, rk, j].astype(self.accum_dtype), axis=rk),
            name="StateProduct",
        )
        local = te.compute(
            (groups, self.chunk_size, self.value_dim),
            lambda g, i, j: te.sum(a[g, i, rs].astype(self.accum_dtype) * v[g, rs, j].astype(self.accum_dtype), axis=rs),
            name="LocalProduct",
        )
        out = te.compute(state.shape, lambda g, i, j: state[g, i, j] + local[g, i, j], name="O")
        if self.out_dtype != self.accum_dtype:
            out = te.compute(out.shape, lambda g, i, j: out[g, i, j].astype(self.out_dtype), name="D")
        self.set_function(te.create_prim_func([q, hidden, a, v, out]))

    def get_hardware_aware_configs(self, arch=None, topk=10):
        return get_roller_hints_from_func(self._func, arch=arch, topk=topk, allow_gemv=False)

    def params_as_dict(self):
        return {
            "batch_size": self.batch_size,
            "num_heads": self.num_heads,
            "sequence": self.sequence,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "chunk_size": self.chunk_size,
            "in_dtype": self.in_dtype,
            "out_dtype": self.out_dtype,
            "accum_dtype": self.accum_dtype,
        }

    @property
    def class_attributes(self):
        return self.params_as_dict()
