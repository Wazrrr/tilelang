"""KDA chunk output: gated Q @ state + causal A @ V.

The leading dimension enumerates independent (batch, chunk, head) groups.
Both input-dtype rounding points in the gated query are preserved.
"""

from dataclasses import dataclass

from tvm import te, tirx

from .base import BaseTemplate
from .graph import TemplateGraph
from ..utils import get_roller_hints_from_output_nodes


@dataclass
class KDAChunkOutputTemplate(BaseTemplate):
    batch_size: int = 1
    num_heads: int = 1
    seq_length: int = 128
    head_dim: int = 64
    value_dim: int = 64
    chunk_size: int = 64
    in_dtype: str = "float16"
    out_dtype: str = "float16"
    accum_dtype: str = "float32"

    def initialize_function(self):
        s, dk, dv = self.chunk_size, self.head_dim, self.value_dim
        if min(self.batch_size, self.num_heads, self.seq_length, s, dk, dv) <= 0 or self.seq_length % s:
            raise ValueError("KDA requires positive dimensions and complete chunks")
        b = self.batch_size * self.num_heads * (self.seq_length // s)
        graph = TemplateGraph()
        q = graph.input("Q", (b, s, dk), self.in_dtype)
        v = graph.input("V", (b, s, dv), self.in_dtype)
        g = graph.input("G", (b, s, dk), "float32")
        a = graph.input("A", (b, s, s), self.in_dtype)
        state = graph.input("State", (b, dk, dv), self.in_dtype)
        scaled = graph.stage(
            "scale_query",
            [q],
            lambda q: te.compute(
                (b, s, dk), lambda h, i, j: (q[h, i, j].astype("float32") * dk**-0.5).astype(self.in_dtype), name="ScaledQ"
            ),
        )
        gated = graph.stage(
            "gate_query",
            [scaled, g],
            lambda q, g: te.compute(
                (b, s, dk), lambda h, i, j: (q[h, i, j].astype("float32") * tirx.exp2(g[h, i, j])).astype(self.in_dtype), name="GatedQ"
            ),
        )
        rk = te.reduce_axis((0, dk), "key_k")
        carried = graph.stage(
            "query_state",
            [gated, state],
            lambda q, h: te.compute(
                (b, s, dv),
                lambda b, i, j: te.sum(q[b, i, rk].astype(self.accum_dtype) * h[b, rk, j].astype(self.accum_dtype), rk),
                name="Carried",
            ),
            tensorcore=True,
        )
        masked = graph.stage(
            "causal_mask",
            [a],
            lambda a: te.compute(
                (b, s, s), lambda h, i, j: tirx.if_then_else(i >= j, a[h, i, j], tirx.const(0, self.in_dtype)), name="MaskedA"
            ),
        )
        rs = te.reduce_axis((0, s), "chunk_k")
        local = graph.stage(
            "attention_value",
            [masked, v],
            lambda a, v: te.compute(
                (b, s, dv),
                lambda h, i, j: te.sum(a[h, i, rs].astype(self.accum_dtype) * v[h, rs, j].astype(self.accum_dtype), rs),
                name="Local",
            ),
            tensorcore=True,
        )
        output = graph.stage(
            "output",
            [carried, local],
            lambda x, y: te.compute((b, s, dv), lambda h, i, j: (x[h, i, j] + y[h, i, j]).astype(self.out_dtype), name="Output"),
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
