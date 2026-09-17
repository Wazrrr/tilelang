"""Build one functional TE graph and the equivalent connected Carver graph."""

from tvm import te

from ..roller import Edge, OutputNode, PrimFuncNode
from ..utils import get_tensorized_func_and_tags


class TemplateGraph:
    """Small graph builder shared by fused multi-stage Carver templates."""

    def __init__(self):
        self.inputs = []
        self.stages = []

    def input(self, name, shape, dtype):
        tensor = te.placeholder(shape, name=name, dtype=dtype)
        self.inputs.append(tensor)
        return tensor

    def stage(self, name, inputs, compute, *, tensorcore=False):
        output = compute(*inputs)
        placeholders = [te.placeholder(t.shape, dtype=t.dtype, name=t.op.name) for t in inputs]
        local_output = compute(*placeholders)
        function = te.create_prim_func([*placeholders, local_output])
        self.stages.append((name, inputs, output, function, tensorcore))
        return output

    def function(self, output):
        return te.create_prim_func([*self.inputs, output])

    def output_nodes(self, output, arch):
        if arch is None:
            raise ValueError("Carver graph construction requires an explicit architecture")
        nodes = {}
        for name, inputs, tensor, function, tensorcore in self.stages:
            tags = {}
            if tensorcore:
                function, tags = get_tensorized_func_and_tags(function, arch.target, allow_gemv=True)
                if not tags:
                    raise ValueError(f"Carver cannot tensorize template stage {name}")
            node = PrimFuncNode(function, name=name, tags=tags)
            nodes[tensor] = node
            # Tensorization may reorder parameters. Connect graph edges by the
            # stable TE input name rather than by the pre-transform position.
            by_name = {t.op.name: t for t in inputs}
            for index, buffer in enumerate(node.input_buffers):
                tensor_input = by_name[buffer.name]
                if tensor_input in nodes:
                    source = nodes[tensor_input]
                    edge = Edge(source, node, 0, index)
                    source._out_edges.append(edge)
                    node.set_inputs(index, edge)
        return [OutputNode(nodes[output])]
