"""Chunk KDA output: gated Q @ H plus causal A @ V, including casts."""

from dataclasses import dataclass, field
from tvm import te, tirx
from .base import BaseTemplate


@dataclass
class KDAChunkTemplate(BaseTemplate):
    _arch: object = field(default=None, repr=False)
    batch: int = 1
    heads: int = 1
    sequence: int = 64
    dim: int = 64
    value_dim: int = 64
    chunk_size: int = 64
    in_dtype: str = "float16"
    out_dtype: str = "float16"
    accum_dtype: str = "float32"

    def initialize_function(self):
        dimensions = (self.batch, self.heads, self.sequence, self.dim, self.value_dim, self.chunk_size)
        if any(type(n) is not int or n <= 0 for n in dimensions) or self.sequence % self.chunk_size:
            raise ValueError("KDA requires positive dimensions and complete chunks")
        groups = self.batch * self.heads * (self.sequence // self.chunk_size)
        s, k, v = self.chunk_size, self.dim, self.value_dim
        q = te.placeholder((groups, s, k), self.in_dtype, name="Q")
        gate = te.placeholder((groups, s, k), "float32", name="Gate")
        h = te.placeholder((groups, k, v), self.in_dtype, name="H")
        a = te.placeholder((groups, s, s), self.in_dtype, name="A")
        values = te.placeholder((groups, s, v), self.in_dtype, name="V")
        scaled = te.compute(q.shape, lambda b, i, j: (q[b, i, j].astype("float32") * k**-0.5).astype(self.in_dtype), name="ScaledQ")
        gated = te.compute(
            q.shape, lambda b, i, j: (scaled[b, i, j].astype("float32") * tirx.exp2(gate[b, i, j])).astype(self.in_dtype), name="GatedQ"
        )
        causal = te.compute(a.shape, lambda b, i, j: tirx.if_then_else(i >= j, a[b, i, j], tirx.const(0, self.in_dtype)), name="CausalA")
        rk, rs = te.reduce_axis((0, k), name="rk"), te.reduce_axis((0, s), name="rs")
        state = te.compute(
            (groups, s, v),
            lambda b, i, j: te.sum(gated[b, i, rk].astype(self.accum_dtype) * h[b, rk, j].astype(self.accum_dtype), axis=rk),
            name="State",
        )
        local = te.compute(
            (groups, s, v),
            lambda b, i, j: te.sum(causal[b, i, rs].astype(self.accum_dtype) * values[b, rs, j].astype(self.accum_dtype), axis=rs),
            name="Local",
        )
        out = te.compute((groups, s, v), lambda b, i, j: (state[b, i, j] + local[b, i, j]).astype(self.out_dtype), name="Output")
        self.set_function(te.create_prim_func([q, gate, h, a, values, out]))
        # Carver accepts one reduction per PrimFunc. Keep both stage graphs
        # and connect their outputs to the fused addition/cast node.
        self.stages = [te.create_prim_func([q, gate, h, state]), te.create_prim_func([a, values, local])]
        if self.arch is not None:
            from ..matmul_analysis import get_tensorized_func_and_tags
            from ..roller import PrimFuncNode, OutputNode, Edge

            nodes = []
            for name, func in zip(("gated_state", "causal_local"), self.stages):
                normalized, tags = get_tensorized_func_and_tags(func, self.arch.target, allow_gemv=True)
                if normalized is None or not tags:
                    raise ValueError(f"Carver cannot tensorize the KDA {name} stage")
                nodes.append(PrimFuncNode(normalized, name=name, tags=tags))
            state_in = te.placeholder((groups, s, v), self.accum_dtype, name="StateInput")
            local_in = te.placeholder((groups, s, v), self.accum_dtype, name="LocalInput")
            combined = te.compute(
                (groups, s, v), lambda b, i, j: (state_in[b, i, j] + local_in[b, i, j]).astype(self.out_dtype), name="Combined"
            )
            terminal = PrimFuncNode(te.create_prim_func([state_in, local_in, combined]), name="output")
            for i, source in enumerate(nodes):
                edge = Edge(source, terminal, 0, i)
                source._out_edges.append(edge)
                terminal.set_inputs(i, edge)
            self.stage_nodes = nodes
            self.set_output_nodes([OutputNode(terminal)])
            self._graph_arch = self.arch

    def get_hardware_aware_configs(self, arch=None, topk=10):
        from ..utils import get_roller_hints_from_output_nodes

        arch = arch or self.arch
        if arch is None:
            raise ValueError("KDA hardware-aware configs require an explicit architecture")
        if getattr(self, "_graph_arch", None) is not arch:
            self.with_arch(arch)
            self.initialize_function()
        return get_roller_hints_from_output_nodes(self.output_nodes, arch=arch, topk=topk)
