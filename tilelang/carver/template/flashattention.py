"""Canonical Carver graph for stable scaled attention."""

from dataclasses import dataclass

from tvm import te, tirx

from .base import BaseTemplate
from .graph import TemplateGraph
from ..utils import get_roller_hints_from_output_nodes


@dataclass
class FlashAttentionTemplate(BaseTemplate):
    """QK, masking, softmax and PV as one connected Carver graph."""

    batch_size: int = 1
    num_heads: int = 1
    head_dim: int = 64
    seq_length: int = 128
    seq_kv_length: int = 128
    is_causal: bool = False
    in_dtype: str = "float16"
    out_dtype: str = "float16"
    accum_dtype: str = "float32"

    def initialize_function(self) -> None:
        if str(self.in_dtype) not in ("float16", "bfloat16") or str(self.accum_dtype) not in ("float16", "float32"):
            raise ValueError("attention requires FP16/BF16 inputs and floating-point accumulation")
        batch_heads = self.batch_size * self.num_heads
        query_length, key_length, head_dim = self.seq_length, self.seq_kv_length, self.head_dim
        if min(batch_heads, query_length, key_length, head_dim) <= 0:
            raise ValueError("attention dimensions must be positive")

        graph = TemplateGraph()
        query = graph.input("Q", (batch_heads, query_length, head_dim), self.in_dtype)
        key = graph.input("K", (batch_heads, key_length, head_dim), self.in_dtype)
        value = graph.input("V", (batch_heads, key_length, head_dim), self.in_dtype)

        reduction_head = te.reduce_axis((0, head_dim), "head_k")
        scores = graph.stage(
            "qk",
            [query, key],
            lambda q, k: te.compute(
                (batch_heads, query_length, key_length),
                lambda b, i, j: te.sum(
                    q[b, i, reduction_head].astype(self.accum_dtype)
                    * k[b, j, reduction_head].astype(self.accum_dtype),
                    reduction_head,
                ),
                name="Scores",
            ),
            tensorcore=True,
        )
        scores = graph.stage(
            "scale_mask",
            [scores],
            lambda source: te.compute(
                (batch_heads, query_length, key_length),
                lambda b, i, j: (
                    tirx.if_then_else(
                        i >= j,
                        source[b, i, j] * head_dim**-0.5,
                        tirx.const(float("-inf"), self.accum_dtype),
                    )
                    if self.is_causal
                    else source[b, i, j] * head_dim**-0.5
                ),
                name="Scaled",
            ),
        )
        reduction_max = te.reduce_axis((0, key_length), "max_k")
        maximum = graph.stage(
            "row_max",
            [scores],
            lambda source: te.compute(
                (batch_heads, query_length),
                lambda b, i: te.max(source[b, i, reduction_max], reduction_max),
                name="Maximum",
            ),
        )
        exponentials = graph.stage(
            "exp",
            [scores, maximum],
            lambda source, row_max: te.compute(
                (batch_heads, query_length, key_length),
                lambda b, i, j: te.exp(source[b, i, j] - row_max[b, i]),
                name="Exponentials",
            ),
        )
        reduction_sum = te.reduce_axis((0, key_length), "sum_k")
        denominator = graph.stage(
            "row_sum",
            [exponentials],
            lambda source: te.compute(
                (batch_heads, query_length),
                lambda b, i: te.sum(source[b, i, reduction_sum], reduction_sum),
                name="Denominator",
            ),
        )
        probabilities = graph.stage(
            "probability_cast",
            [exponentials],
            lambda source: te.compute(
                (batch_heads, query_length, key_length),
                lambda b, i, j: source[b, i, j].astype(self.in_dtype),
                name="Probabilities",
            ),
        )
        reduction_sequence = te.reduce_axis((0, key_length), "sequence_k")
        numerator = graph.stage(
            "pv",
            [probabilities, value],
            lambda probability, values: te.compute(
                (batch_heads, query_length, head_dim),
                lambda b, i, j: te.sum(
                    probability[b, i, reduction_sequence].astype(self.accum_dtype)
                    * values[b, reduction_sequence, j].astype(self.accum_dtype),
                    reduction_sequence,
                ),
                name="Numerator",
            ),
            tensorcore=True,
        )
        output = graph.stage(
            "normalize",
            [numerator, denominator],
            lambda source, normalizer: te.compute(
                (batch_heads, query_length, head_dim),
                lambda b, i, j: (source[b, i, j] / normalizer[b, i]).astype(self.out_dtype),
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
