"""Canonical Carver graph for the KDA chunk-output kernel."""

from dataclasses import dataclass

from tvm import te, tirx

from .base import BaseTemplate
from .graph import TemplateGraph
from ..utils import get_roller_hints_from_output_nodes


@dataclass
class KDAChunkTemplate(BaseTemplate):
    """Gated Q@state plus causal A@V, including both input casts."""

    batch_size: int = 1
    num_heads: int = 1
    sequence: int = 128
    key_dim: int = 64
    value_dim: int = 64
    chunk_size: int = 64
    in_dtype: str = "float16"
    out_dtype: str = "float16"
    accum_dtype: str = "float32"

    def initialize_function(self) -> None:
        chunk, key_dim, value_dim = self.chunk_size, self.key_dim, self.value_dim
        dimensions = (self.batch_size, self.num_heads, self.sequence, chunk, key_dim, value_dim)
        if any(type(value) is not int or value <= 0 for value in dimensions) or self.sequence % chunk:
            raise ValueError("KDA dimensions must be positive and sequence must contain complete chunks")
        groups = self.batch_size * self.num_heads * (self.sequence // chunk)

        graph = TemplateGraph()
        query = graph.input("Q", (groups, chunk, key_dim), self.in_dtype)
        value = graph.input("V", (groups, chunk, value_dim), self.in_dtype)
        gate = graph.input("G", (groups, chunk, key_dim), "float32")
        attention = graph.input("A", (groups, chunk, chunk), self.in_dtype)
        hidden = graph.input("State", (groups, key_dim, value_dim), self.in_dtype)

        scaled = graph.stage(
            "scale_query",
            [query],
            lambda source: te.compute(
                (groups, chunk, key_dim),
                lambda b, i, j: (source[b, i, j].astype("float32") * key_dim**-0.5).astype(self.in_dtype),
                name="ScaledQ",
            ),
        )
        gated = graph.stage(
            "gate_query",
            [scaled, gate],
            lambda source, gates: te.compute(
                (groups, chunk, key_dim),
                lambda b, i, j: (source[b, i, j].astype("float32") * tirx.exp2(gates[b, i, j])).astype(self.in_dtype),
                name="GatedQ",
            ),
        )
        reduction_key = te.reduce_axis((0, key_dim), "key_k")
        carried = graph.stage(
            "query_state",
            [gated, hidden],
            lambda source, state: te.compute(
                (groups, chunk, value_dim),
                lambda b, i, j: te.sum(
                    source[b, i, reduction_key].astype(self.accum_dtype)
                    * state[b, reduction_key, j].astype(self.accum_dtype),
                    reduction_key,
                ),
                name="Carried",
            ),
            tensorcore=True,
        )
        masked = graph.stage(
            "causal_mask",
            [attention],
            lambda source: te.compute(
                (groups, chunk, chunk),
                lambda b, i, j: tirx.if_then_else(i >= j, source[b, i, j], tirx.const(0, self.in_dtype)),
                name="MaskedA",
            ),
        )
        reduction_chunk = te.reduce_axis((0, chunk), "chunk_k")
        local = graph.stage(
            "attention_value",
            [masked, value],
            lambda weights, values: te.compute(
                (groups, chunk, value_dim),
                lambda b, i, j: te.sum(
                    weights[b, i, reduction_chunk].astype(self.accum_dtype)
                    * values[b, reduction_chunk, j].astype(self.accum_dtype),
                    reduction_chunk,
                ),
                name="Local",
            ),
            tensorcore=True,
        )
        output = graph.stage(
            "output",
            [carried, local],
            lambda state, local_value: te.compute(
                (groups, chunk, value_dim),
                lambda b, i, j: (state[b, i, j] + local_value[b, i, j]).astype(self.out_dtype),
                name="Output",
            ),
        )
        self._graph, self._output = graph, output
        self.set_function(graph.function(output))

    @property
    def output_nodes(self):
        return self._graph.output_nodes(self._output, self.arch)

    def get_hardware_aware_configs(self, arch=None, topk=10):
        selected_arch = arch or self.arch
        return get_roller_hints_from_output_nodes(
            self._graph.output_nodes(self._output, selected_arch), arch=selected_arch, topk=topk
        )

    def params_as_dict(self):
        return {
            name: getattr(self, name)
            for name in (
                "batch_size",
                "num_heads",
                "sequence",
                "key_dim",
                "value_dim",
                "chunk_size",
                "in_dtype",
                "out_dtype",
                "accum_dtype",
            )
        }

    @property
    def class_attributes(self):
        return self.params_as_dict()
