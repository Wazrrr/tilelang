"""Stable scaled attention, including the mask and probability conversion.

Carver represents the fused tensor graph; TileLang's online loop scheduling is
an implementation detail and is modeled separately by TileTune.
"""

from dataclasses import dataclass

from tvm import te, tirx

from .base import BaseTemplate
from .graph import TemplateGraph
from ..utils import get_roller_hints_from_output_nodes


@dataclass
class FlashAttentionTemplate(BaseTemplate):
    batch_size: int = 1
    num_heads: int = 1
    head_dim: int = 64
    seq_length: int = 128
    seq_kv_length: int = 128
    is_causal: bool = False
    in_dtype: str = "float16"
    out_dtype: str = "float16"
    accum_dtype: str = "float32"

    def initialize_function(self):
        if str(self.in_dtype) not in ("float16", "bfloat16") or str(self.accum_dtype) not in ("float16", "float32"):
            raise ValueError("attention requires FP16/BF16 inputs and floating-point accumulation")
        b, m, n, d = self.batch_size * self.num_heads, self.seq_length, self.seq_kv_length, self.head_dim
        if min(b, m, n, d) <= 0:
            raise ValueError("attention dimensions must be positive")
        graph = TemplateGraph()
        q = graph.input("Q", (b, m, d), self.in_dtype)
        k = graph.input("K", (b, n, d), self.in_dtype)
        v = graph.input("V", (b, n, d), self.in_dtype)
        rk = te.reduce_axis((0, d), "head_k")
        scores = graph.stage(
            "qk",
            [q, k],
            lambda q, k: te.compute(
                (b, m, n),
                lambda h, i, j: te.sum(q[h, i, rk].astype(self.accum_dtype) * k[h, j, rk].astype(self.accum_dtype), rk),
                name="Scores",
            ),
            tensorcore=True,
        )
        scores = graph.stage(
            "scale_mask",
            [scores],
            lambda s: te.compute(
                (b, m, n),
                lambda h, i, j: (
                    tirx.if_then_else(i >= j, s[h, i, j] * d**-0.5, tirx.const(float("-inf"), self.accum_dtype))
                    if self.is_causal
                    else s[h, i, j] * d**-0.5
                ),
                name="Scaled",
            ),
        )
        rm = te.reduce_axis((0, n), "max_k")
        maximum = graph.stage("row_max", [scores], lambda s: te.compute((b, m), lambda h, i: te.max(s[h, i, rm], rm), name="Maximum"))
        exponentials = graph.stage(
            "exp",
            [scores, maximum],
            lambda s, mx: te.compute((b, m, n), lambda h, i, j: te.exp(s[h, i, j] - mx[h, i]), name="Exponentials"),
        )
        rs = te.reduce_axis((0, n), "sum_k")
        denominator = graph.stage(
            "row_sum", [exponentials], lambda e: te.compute((b, m), lambda h, i: te.sum(e[h, i, rs], rs), name="Denominator")
        )
        probabilities = graph.stage(
            "probability_cast",
            [exponentials],
            lambda e: te.compute((b, m, n), lambda h, i, j: e[h, i, j].astype(self.in_dtype), name="Probabilities"),
        )
        rv = te.reduce_axis((0, n), "sequence_k")
        numerator = graph.stage(
            "pv",
            [probabilities, v],
            lambda p, v: te.compute(
                (b, m, d),
                lambda h, i, j: te.sum(p[h, i, rv].astype(self.accum_dtype) * v[h, rv, j].astype(self.accum_dtype), rv),
                name="Numerator",
            ),
            tensorcore=True,
        )
        output = graph.stage(
            "normalize",
            [numerator, denominator],
            lambda o, l: te.compute((b, m, d), lambda h, i, j: (o[h, i, j] / l[h, i]).astype(self.out_dtype), name="Output"),
        )
        self._graph, self._output = graph, output
        self.set_function(graph.function(output))

    @property
    def output_nodes(self):
        return self._graph.output_nodes(self._output, self.arch)

    def get_hardware_aware_configs(self, arch=None, topk=10):
        return get_roller_hints_from_output_nodes(
            self._graph.output_nodes(self._output, arch or self.arch), arch=arch or self.arch, topk=topk
        )

    def params_as_dict(self):
        return {
            name: getattr(self, name)
            for name in (
                "batch_size",
                "num_heads",
                "head_dim",
                "seq_length",
                "seq_kv_length",
                "is_causal",
                "in_dtype",
                "out_dtype",
                "accum_dtype",
            )
        }

    @property
    def class_attributes(self):
        return self.params_as_dict()
