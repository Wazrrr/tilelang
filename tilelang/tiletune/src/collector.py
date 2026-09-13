"""Capture native operators, scalar accesses and reaching dependencies."""

import tvm_ffi
from tilelang import tvm
from tvm import tirx as tir
from tvm.ir import Range
from .ir import Operation, Region
from .ir_utils import _int, _exclusive
from .regions import _contains, _intersect


def _opaque_call(node):
    if not isinstance(node, tir.Call):
        return False
    # Read registered effects rather than whitelisting softmax intrinsic names.
    effect = node.op.get_attr("TCallEffectKind") if isinstance(node.op, tvm.ir.Op) else None
    return (
        effect is None
        or int(effect) > tir.CallEffectKind.Pure.value
        or str(node.dtype) == "handle"
        or any(str(arg.dtype) == "handle" and not (isinstance(arg, tir.StringImm) and node.op.name == "tl.infinity") for arg in node.args)
    )


class _Collector:
    def __init__(self, func):
        self.operations = []
        self.buffers = list(func.buffer_map.values())
        self.layouts = {}
        self.threads = {}
        self.block_domains = {}
        self.active_threads = {}
        self.active_pipeline_stages = ()
        self.pipeline_loops = []
        self.serial_loops = []
        self.bindings = {}
        self.unknown = []
        self.parse = tvm_ffi.get_global_func("tl.tiletune.ParseOperator")
        self.access = tvm_ffi.get_global_func("tl.tiletune.GetAccessRegions")
        self.visit(func.body)
        for i, buffer in enumerate(self.buffers):
            if any(buffer.data.same_as(other.data) and not buffer.same_as(other) for other in self.buffers[:i]):
                self.unknown.append("multiple buffer views share a data variable")
        # Reaching writers: kill only proven complete, unconditional overwrites.
        reaching = []
        for op in self.operations:
            op.dependencies = sorted(
                {
                    old.index
                    for old in reaching
                    for w in old.writes
                    for r in op.reads
                    if w.buffer.same_as(r.buffer) and _intersect(w, r) is not None and not _exclusive(old, op)
                }
            )
            reaching = [
                old
                for old in reaching
                if not (
                    not op.predicates
                    and not op.loops
                    and old.writes
                    and all(any(_contains(w, prev) for w in op.writes) for prev in old.writes)
                )
            ]
            reaching.append(op)

    def add(self, kind, reads, writes, metadata, loops, predicates, branches, unknown=False):
        op = Operation(len(self.operations), kind, reads, writes, metadata, loops, predicates, branches, unknown=unknown)
        op.launch_threads = dict(self.active_threads)
        op.pipeline_stages = self.active_pipeline_stages
        self.operations.append(op)
        for r in reads + writes:
            if r.buffer not in self.buffers:
                self.buffers.append(r.buffer)
        if unknown:
            self.unknown.append(f"operation {op.index}: {kind}")

    def visit(self, node, loops=(), predicates=(), branches=(), annotations=None):
        annotations = dict(annotations or {})

        def resolve(expr):
            return tir.stmt_functor.substitute(expr, self.bindings) if self.bindings else expr

        def visit(child, **kw):
            self.visit(
                child,
                kw.get("loops", loops),
                kw.get("predicates", predicates),
                kw.get("branches", branches),
                kw.get("annotations", annotations),
            )

        if isinstance(node, tir.SeqStmt):
            for stmt in node.seq:
                visit(stmt)
        elif isinstance(node, tir.Bind):
            value = resolve(node.value)
            unsafe = []
            tir.stmt_functor.post_order_visit(
                value, lambda n: unsafe.append(n) if isinstance(n, tir.BufferLoad) or _opaque_call(n) else None
            )
            if unsafe:
                self.unknown.append("unresolved data-dependent binding")
            else:
                # Substitute pure SSA index expressions in the analysis view;
                # the original PrimFunc is retained unchanged for compilation.
                self.bindings[node.var] = value
        elif isinstance(node, tir.For):
            node = tir.For(
                node.loop_var, resolve(node.min), resolve(node.extent), node.kind, node.body, node.thread_binding, node.annotations
            )
            previous_threads = dict(self.active_threads)
            previous_pipeline = self.active_pipeline_stages
            if node.kind == tir.ForKind.SERIAL and not node.annotations:
                self.serial_loops.append(node)
            if "num_stages" in node.annotations:
                self.pipeline_loops.append(node)
                self.active_pipeline_stages += (_int(node.annotations["num_stages"]),)
            elif "software_pipeline_stage" in node.annotations:
                self.active_pipeline_stages += (None,)
            if node.thread_binding is not None:
                if node.thread_binding.thread_tag.startswith("blockIdx."):
                    self.block_domains[node.thread_binding.thread_tag] = (node.loop_var, Range.from_min_extent(node.min, node.extent))
                self.threads[node.thread_binding.thread_tag] = node.extent
                self.active_threads[node.thread_binding.thread_tag] = node.extent
            visit(node.body, loops=loops + ((node.loop_var, Range.from_min_extent(node.min, node.extent), str(node.kind)),))
            self.active_threads = previous_threads
            self.active_pipeline_stages = previous_pipeline
        elif isinstance(node, tir.AttrStmt):
            previous_threads = dict(self.active_threads)
            if node.attr_key == "thread_extent":
                if node.node.thread_tag.startswith("blockIdx."):
                    self.block_domains[node.node.thread_tag] = (node.node.var, Range.from_min_extent(0, node.value))
                self.threads[node.node.thread_tag] = node.value
                self.active_threads[node.node.thread_tag] = node.value
            visit(node.body)
            self.active_threads = previous_threads
        elif isinstance(node, tir.IfThenElse):
            ident = len(self.operations), id(node)
            condition = resolve(node.condition)
            visit(node.then_case, predicates=predicates + (condition,), branches=branches + ((ident, True),))
            if node.else_case is not None:
                visit(node.else_case, predicates=predicates + (tir.Not(condition),), branches=branches + ((ident, False),))
        elif isinstance(node, tir.SBlockRealize):
            pred = () if _int(node.predicate) == 1 else (node.predicate,)
            visit(node.block, predicates=predicates + pred)
        elif isinstance(node, tir.SBlock):
            annotations.update(dict(node.annotations))
            self.layouts.update(dict(annotations.get("layout_map", {})))
            self.buffers.extend(b for b in node.alloc_buffers if b not in self.buffers)
            if node.match_buffers:
                self.unknown.append("unresolved match-buffer aliases")
            if node.init is not None:
                visit(node.init, annotations=annotations)
            visit(node.body, annotations=annotations)
        elif isinstance(node, tir.BufferStore):
            node = tir.BufferStore(node.buffer, resolve(node.value), [resolve(i) for i in node.indices])
            reads = []
            tir.stmt_functor.post_order_visit(
                node.value,
                lambda n: (
                    reads.append(Region(n.buffer, [Range.from_min_extent(i, 1) for i in n.indices]))
                    if isinstance(n, tir.BufferLoad)
                    else None
                ),
            )
            writes = [Region(node.buffer, [Range.from_min_extent(i, 1) for i in node.indices])]
            opaque = []
            tir.stmt_functor.post_order_visit(node.value, lambda n: opaque.append(n) if _opaque_call(n) else None)
            for index in node.indices:
                tir.stmt_functor.post_order_visit(index, lambda n: opaque.append(n) if isinstance(n, tir.BufferLoad | tir.Call) else None)
            self.add("elementwise", reads, writes, node, loops, predicates, branches, bool(opaque))
        elif isinstance(node, tir.Evaluate) and isinstance(node.value, tir.Call):
            call = resolve(node.value)
            op = self.parse(call, annotations)
            if op is None:
                self.add(str(call.op), [], [], None, loops, predicates, branches, True)
            else:
                reads, writes = self.access(op)
                self.add(
                    str(call.op.name).split(".")[-1],
                    [Region.from_ir(r) for r in reads],
                    [Region.from_ir(r) for r in writes],
                    op,
                    loops,
                    predicates,
                    branches,
                )
        elif hasattr(node, "body"):
            # Let/Bind substitutions and unusual scopes require conservative handling.
            self.unknown.append(f"unmodeled scope: {type(node).__name__}")
            visit(node.body)
        elif not isinstance(node, tir.Evaluate):
            self.unknown.append(f"unmodeled statement: {type(node).__name__}")
