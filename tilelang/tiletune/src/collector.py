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


def _uses_global_buffer(node, buffers):
    """Return whether an opaque expression receives or loads a global buffer."""
    found = False

    def visit(value):
        nonlocal found
        if isinstance(value, tir.BufferLoad) and value.buffer.scope() == "global":
            found = True
        elif isinstance(value, tir.Var) and any(value.same_as(buffer.data) and buffer.scope() == "global" for buffer in buffers):
            found = True

    tir.stmt_functor.post_order_visit(node, visit)
    return found


def _same_scalar_slot(a, b):
    """Return whether two scalar local-buffer accesses name the same slot."""
    return (
        a.buffer.same_as(b.buffer)
        and len(a.indices) == len(b.indices) == 1
        and _int(a.indices[0]) == _int(b.indices[0]) == 0
        and a.buffer.scope().startswith("local")
    )


class _Collector:
    def __init__(self, func, *, track_dependencies=True):
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
        # Latest straight-line values of scalar local buffers.  Eager TileLang
        # lowers mutable scheduler counters to these one-element buffers.
        self.scalar_values = {}
        self.unknown = []
        self.memory_unknown = []
        self.dependencies_tracked = track_dependencies
        self.parse = tvm_ffi.get_global_func("tl.tiletune.ParseOperator")
        self.access = tvm_ffi.get_global_func("tl.tiletune.GetAccessRegions")
        self.visit(func.body)
        for i, buffer in enumerate(self.buffers):
            if any(buffer.data.same_as(other.data) and not buffer.same_as(other) for other in self.buffers[:i]):
                self.unknown.append("multiple buffer views share a data variable")
                self.memory_unknown.append("multiple buffer views share a data variable")
        if not track_dependencies:
            return
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

    def add(self, kind, reads, writes, metadata, loops, predicates, branches, unknown=False, memory_unknown=False):
        op = Operation(len(self.operations), kind, reads, writes, metadata, loops, predicates, branches, unknown=unknown)
        op.launch_threads = dict(self.active_threads)
        op.pipeline_stages = self.active_pipeline_stages
        self.operations.append(op)
        for r in reads + writes:
            if r.buffer not in self.buffers:
                self.buffers.append(r.buffer)
        if unknown:
            reason = f"operation {op.index}: {kind}"
            self.unknown.append(reason)
            if memory_unknown:
                self.memory_unknown.append(reason)

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
                # A loaded scalar may make addresses data-dependent without
                # making memory volume unknown. Region extents, enclosing-loop
                # visits, and the launch grid are validated independently by
                # the memory scorer; any unresolved one still disables the
                # score. Opaque calls remain unknown memory effects.
                opaque = [item for item in unsafe if not isinstance(item, tir.BufferLoad)]
                if any(_uses_global_buffer(item, self.buffers) for item in opaque):
                    self.memory_unknown.append("unresolved data-dependent binding reads global memory")
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
            previous_scalars = dict(self.scalar_values)
            visit(node.then_case, predicates=predicates + (condition,), branches=branches + ((ident, True),))
            self.scalar_values = dict(previous_scalars)
            if node.else_case is not None:
                visit(node.else_case, predicates=predicates + (tir.Not(condition),), branches=branches + ((ident, False),))
            self.scalar_values = previous_scalars
        elif isinstance(node, tir.While):
            # Prove the common lowered form
            #   scalar = nonnegative_start
            #   while scalar < stop: ...; scalar = scalar + positive_step
            # and charge every enclosed access its conservative maximum visit
            # count.  This is derived solely from the PrimFunc; an arbitrary
            # or data-dependent while loop remains unknown.
            condition = resolve(node.condition)
            load = condition.a if isinstance(condition, tir.LT) and isinstance(condition.a, tir.BufferLoad) else None
            stop = _int(condition.b) if load is not None else None
            initial = self.scalar_values.get(load.buffer) if load is not None else None
            updates = []
            if load is not None:
                tir.stmt_functor.post_order_visit(
                    node.body,
                    lambda value: updates.append(value)
                    if isinstance(value, tir.BufferStore) and _same_scalar_slot(value, load)
                    else None,
                )
            step = None
            if len(updates) == 1 and isinstance(updates[0].value, tir.Add):
                terms = (updates[0].value.a, updates[0].value.b)
                increment = next((value for value in terms if not isinstance(value, tir.BufferLoad)), None)
                recurrence = next((value for value in terms if isinstance(value, tir.BufferLoad)), None)
                body_statements = node.body.seq if isinstance(node.body, tir.SeqStmt) else (node.body,)
                unconditional = any(updates[0].same_as(statement) for statement in body_statements)
                if recurrence is not None and _same_scalar_slot(recurrence, load) and unconditional:
                    step = _int(resolve(increment))
            visits = None
            if initial is not None and stop is not None and step is not None and step > 0:
                analyzer = tvm.arith.Analyzer()
                for var, domain in self.block_domains.values():
                    analyzer.bind(var, domain)
                for var, domain, _ in loops:
                    analyzer.bind(var, domain)
                lower = _int(analyzer.const_int_bound(resolve(initial)).min_value)
                if lower is not None and lower >= 0:
                    visits = max(0, (stop - int(lower) + step - 1) // step)
            if visits is None:
                reason = "unmodeled scope: While"
                self.unknown.append(reason)
                self.memory_unknown.append(reason)
                visit(node.body)
            else:
                previous_scalars = dict(self.scalar_values)
                loop_var = tir.Var(f"tiletune_while_{len(self.operations)}", "int32")
                visit(node.body, loops=loops + ((loop_var, Range.from_min_extent(0, visits), "while_bound"),))
                self.scalar_values = previous_scalars
        elif isinstance(node, tir.SBlockRealize):
            pred = () if _int(node.predicate) == 1 else (node.predicate,)
            visit(node.block, predicates=predicates + pred)
        elif isinstance(node, tir.SBlock):
            annotations.update(dict(node.annotations))
            self.layouts.update(dict(annotations.get("layout_map", {})))
            self.buffers.extend(b for b in node.alloc_buffers if b not in self.buffers)
            if node.match_buffers:
                self.unknown.append("unresolved match-buffer aliases")
                self.memory_unknown.append("unresolved match-buffer aliases")
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
            self.add(
                "elementwise",
                reads,
                writes,
                node,
                loops,
                predicates,
                branches,
                bool(opaque),
                memory_unknown=any(_uses_global_buffer(value, self.buffers) for value in opaque),
            )
            if node.buffer.scope().startswith("local") and len(node.indices) == 1 and _int(node.indices[0]) == 0:
                self.scalar_values[node.buffer] = node.value
        elif isinstance(node, tir.Evaluate) and isinstance(node.value, tir.Call):
            call = resolve(node.value)
            op = self.parse(call, annotations)
            if op is None:
                name = call.op.name if isinstance(call.op, tvm.ir.Op) else str(call.op)
                # TCGen05 completion is represented by an explicit mbarrier
                # wait. Keep it in program order without making the kernel
                # opaque; synchronization cost is modeled by the pipeline.
                if name == "tl.mbarrier_wait_parity":
                    self.add("barrier", [], [], None, loops, predicates, branches)
                else:
                    self.add(
                        name,
                        [],
                        [],
                        call,
                        loops,
                        predicates,
                        branches,
                        True,
                        memory_unknown=_uses_global_buffer(call, self.buffers),
                    )
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
            reason = f"unmodeled scope: {type(node).__name__}"
            self.unknown.append(reason)
            self.memory_unknown.append(reason)
            visit(node.body)
        elif not isinstance(node, tir.Evaluate):
            reason = f"unmodeled statement: {type(node).__name__}"
            self.unknown.append(reason)
            self.memory_unknown.append(reason)
