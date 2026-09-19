"""Resolve declared integer metadata in the analysis view, keeping the kernel intact."""

from tvm import tirx as tir
from tvm.arith import Analyzer
from .ir_utils import _int


def parameter_values(func, values):
    result = {}
    for index, items in (values or {}).items():
        index = int(index)
        if index < 0 or index >= len(func.params) or func.params[index] not in func.buffer_map:
            raise ValueError("input_values requires a buffer parameter")
        buffer = func.buffer_map[func.params[index]]
        if len(buffer.shape) != 1 or _int(buffer.shape[0]) != len(items) or str(buffer.dtype) not in ("int32", "int64"):
            raise ValueError("input_values shape/dtype does not match the integer parameter")
        if buffer in result:
            raise ValueError("input_values contains duplicate parameter indices")
        bits = int(str(buffer.dtype)[3:])
        if any(not -(2 ** (bits - 1)) <= value < 2 ** (bits - 1) for value in items):
            raise ValueError("input_values exceeds the integer parameter range")
        result[buffer] = tuple(items)

    def check(node):
        if isinstance(node, tir.BufferStore) and node.buffer in result:
            raise ValueError("input_values parameters must be read-only")

    tir.stmt_functor.post_order_visit(func.body, check)
    return result


@tir.functor.mutator
class ValueResolver(tir.PyStmtExprMutator):
    def __init__(self, collector, *, simplify_values=True):
        super().__init__()
        self.col = collector
        self.simplify_values = simplify_values

    def visit_var_(self, node):
        return self.col.bindings.get(node, node)

    def visit_buffer_load_(self, node):
        indices = [self.visit_expr(index) for index in node.indices]
        if node.buffer in self.col.scalar_values and len(indices) == 1 and _int(indices[0]) == 0:
            return self.col.scalar_values[node.buffer]
        values = self.col.input_values.get(node.buffer)
        if values is None:
            return tir.BufferLoad(node.buffer, indices)
        index = indices[0]
        ana = Analyzer()
        for var, domain in self.col.block_domains.values():
            ana.bind(var, domain)
        if not self.simplify_values:
            # Normalize only the lookup index: this is needed to prove the
            # metadata contract, unlike the resulting piecewise address.
            index = ana.simplify(index)
        if not ana.can_prove(index >= 0) or not ana.can_prove(index < len(values)):
            return tir.BufferLoad(node.buffer, indices)
        result = tir.const(values[-1], node.dtype)
        for i in reversed(range(len(values) - 1)):
            result = tir.Select(index == i, tir.const(values[i], node.dtype), result)
        return ana.simplify(result) if self.simplify_values else result

    def visit_call_(self, node):
        args = [self.visit_expr(arg) for arg in node.args]
        if getattr(node.op, "name", None) == "tirx.if_then_else" and str(node.dtype).startswith("int"):
            return tir.Select(*args)
        return tir.Call(node.dtype, node.op, args)


def metadata_loop(node):
    """Bounded scalar setup only; tensor loops and pipelines stay symbolic."""
    if node.kind != tir.ForKind.SERIAL or node.annotations or not 0 < (_int(node.extent) or 0) <= 64:
        return False
    nodes = []
    tir.stmt_functor.post_order_visit(node.body, nodes.append)
    stores = [n for n in nodes if isinstance(n, tir.BufferStore)]
    return (
        bool(stores)
        and all(n.buffer.scope() == "local.var" for n in stores)
        and not any(isinstance(n, tir.For | tir.IfThenElse) or isinstance(n, tir.Evaluate) and isinstance(n.value, tir.Call) for n in nodes)
    )
