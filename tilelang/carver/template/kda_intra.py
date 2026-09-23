"""Canonical Carver graph for token-parallel KDA intra coefficients."""

from dataclasses import dataclass

from tvm import te, tirx

from .base import BaseTemplate
from ..utils import get_roller_hints_from_func


@dataclass
class KDAIntraTemplate(BaseTemplate):
    """Causal query/key and beta-weighted key/key coefficient reductions."""

    batch_size: int = 1
    num_heads: int = 1
    sequence: int = 128
    key_dim: int = 64
    chunk_size: int = 64
    sub_chunk_size: int = 16
    in_dtype: str = "float16"
    out_dtype: str = "float16"
    accum_dtype: str = "float32"
    gate_dtype: str = "float32"

    def initialize_function(self) -> None:
        dimensions = (
            self.batch_size,
            self.num_heads,
            self.sequence,
            self.key_dim,
            self.chunk_size,
            self.sub_chunk_size,
        )
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("KDA dimensions must be positive integers")
        if self.sequence % self.chunk_size or self.chunk_size % self.sub_chunk_size:
            raise ValueError("KDA sequence and chunk dimensions must be complete")

        shape = (self.batch_size, self.sequence, self.num_heads, self.key_dim)
        query = te.placeholder(shape, name="Q", dtype=self.in_dtype)
        key = te.placeholder(shape, name="K", dtype=self.in_dtype)
        gate = te.placeholder(shape, name="GK", dtype=self.gate_dtype)
        beta = te.placeholder(shape[:-1], name="Beta", dtype=self.in_dtype)
        scale = self.key_dim**-0.5

        aqk_reduction = te.reduce_axis((0, self.key_dim), "aqk_k")

        def aqk_reduction_value(batch, token, head, offset):
            source_token = token // self.chunk_size * self.chunk_size + offset
            return te.sum(
                (query[batch, token, head, aqk_reduction].astype(self.accum_dtype) * scale)
                * key[batch, source_token, head, aqk_reduction].astype(self.accum_dtype)
                * tirx.exp2(
                    gate[batch, token, head, aqk_reduction].astype(self.accum_dtype)
                    - gate[batch, source_token, head, aqk_reduction].astype(self.accum_dtype)
                ),
                axis=aqk_reduction,
            )

        aqk_reduced = te.compute(
            (self.batch_size, self.sequence, self.num_heads, self.chunk_size),
            aqk_reduction_value,
            name="AqkReduction",
        )
        aqk = te.compute(
            aqk_reduced.shape,
            lambda batch, token, head, offset: tirx.if_then_else(
                token // self.chunk_size * self.chunk_size + offset <= token,
                aqk_reduced[batch, token, head, offset].astype(self.out_dtype),
                tirx.const(0, self.out_dtype),
            ),
            name="Aqk",
        )

        akk_reduction = te.reduce_axis((0, self.key_dim), "akk_k")

        def akk_reduction_value(batch, token, head, offset):
            source_token = token // self.sub_chunk_size * self.sub_chunk_size + offset
            return te.sum(
                key[batch, token, head, akk_reduction].astype(self.accum_dtype)
                * beta[batch, token, head].astype(self.accum_dtype)
                * key[batch, source_token, head, akk_reduction].astype(self.accum_dtype)
                * tirx.exp2(
                    gate[batch, token, head, akk_reduction].astype(self.accum_dtype)
                    - gate[batch, source_token, head, akk_reduction].astype(self.accum_dtype)
                ),
                axis=akk_reduction,
            )

        akk_reduced = te.compute(
            (self.batch_size, self.sequence, self.num_heads, self.sub_chunk_size),
            akk_reduction_value,
            name="AkkReduction",
        )
        akk = te.compute(
            akk_reduced.shape,
            lambda batch, token, head, offset: tirx.if_then_else(
                token // self.sub_chunk_size * self.sub_chunk_size + offset < token,
                akk_reduced[batch, token, head, offset].astype(self.out_dtype),
                tirx.const(0, self.out_dtype),
            ),
            name="Akk",
        )
        self.set_function(te.create_prim_func([query, key, gate, beta, aqk, akk]))

    def get_hardware_aware_configs(self, arch=None, topk=10):
        return get_roller_hints_from_func(
            self.equivalent_function(),
            arch=arch or self.arch,
            topk=topk,
            allow_gemv=True,
        )

    def params_as_dict(self):
        return {
            name: getattr(self, name)
            for name in (
                "batch_size",
                "num_heads",
                "sequence",
                "key_dim",
                "chunk_size",
                "sub_chunk_size",
                "in_dtype",
                "out_dtype",
                "accum_dtype",
                "gate_dtype",
            )
        }

    @property
    def class_attributes(self):
        return self.params_as_dict()
