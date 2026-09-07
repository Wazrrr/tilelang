"""Read-only tile demand propagation on the original TIRX PrimFunc.

Regions use buffer identity, symbolic offsets and extents. Loop coverage is a
separate conservative union; it never scales the register working set. The
physical model bounds dense accumulator tile state, not compiler scratch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import tvm_ffi
from tilelang import tvm
from tvm import tirx as tir
from tvm.arith import Analyzer
from tvm.ir import Range

from .config import CarverConfig
from .register_pressure import analyze_register_pressure as _pressure


def _int(expr):
    try:
        return int(Analyzer().simplify(expr))
    except (TypeError, ValueError):
        return None


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


@dataclass
class Region:
    buffer: object
    ranges: list
    precision: str = "exact"

    @classmethod
    def from_ir(cls, region, precision="exact"):
        return cls(region.buffer, list(region.region), precision)

    def to_dict(self):
        return {
            "buffer": self.buffer.name,
            "buffer_id": str(hash(self.buffer)),
            "scope": self.buffer.scope(),
            "ranges": [{"min": str(r.min), "extent": str(r.extent)} for r in self.ranges],
            "precision": self.precision,
        }


@dataclass
class Operation:
    index: int
    kind: str
    reads: list[Region]
    writes: list[Region]
    metadata: object = None
    loops: tuple = ()
    predicates: tuple = ()
    branches: tuple = ()
    dependencies: list[int] = field(default_factory=list)
    unknown: bool = False
    demands: list[Region] = field(default_factory=list)
    launch_threads: dict = field(default_factory=dict)
    pipeline_stages: tuple = ()

    def to_dict(self):
        return {
            "index": self.index,
            "kind": self.kind,
            "reads": [r.to_dict() for r in self.reads],
            "writes": [r.to_dict() for r in self.writes],
            "dependencies": self.dependencies,
            "loops": [{"var": str(v), "min": str(r.min), "extent": str(r.extent), "kind": k} for v, r, k in self.loops],
            "predicates": [str(p) for p in self.predicates],
            "unknown": self.unknown,
        }


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
        self.parse = tvm_ffi.get_global_func("tl.new_carver.ParseOperator")
        self.access = tvm_ffi.get_global_func("tl.new_carver.GetAccessRegions")
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
            try:
                op = self.parse(call, annotations)
                if op is None:
                    self.add(str(call.op), [], [], None, loops, predicates, branches, True)
                else:
                    reads, writes = self.access(op)
                    self.add(
                        op.__class__.__name__ if hasattr(op, "__class__") else str(call.op),
                        [Region.from_ir(r) for r in reads],
                        [Region.from_ir(r) for r in writes],
                        op,
                        loops,
                        predicates,
                        branches,
                    )
                    # Generic FFI objects have no useful Python subclass name.
                    self.operations[-1].kind = str(call.op.name).split(".")[-1]
            except Exception as error:
                self.add(str(call.op), [], [], None, loops, predicates, branches, True)
                self.unknown.append(str(error))
        elif hasattr(node, "body"):
            # Let/Bind substitutions and unusual scopes require conservative handling.
            self.unknown.append(f"unmodeled scope: {type(node).__name__}")
            visit(node.body)
        elif not isinstance(node, tir.Evaluate):
            self.unknown.append(f"unmodeled statement: {type(node).__name__}")


def _exclusive(a, b):
    return any(i == j and x != y for i, x in a.branches for j, y in b.branches)


def _contains(outer, inner):
    if not outer.buffer.same_as(inner.buffer) or len(outer.ranges) != len(inner.ranges):
        return False
    ana = Analyzer()
    return all(
        ana.can_prove(a.min <= b.min) and ana.can_prove(a.min + a.extent >= b.min + b.extent) for a, b in zip(outer.ranges, inner.ranges)
    )


def _intersect(a, b):
    if not a.buffer.same_as(b.buffer) or len(a.ranges) != len(b.ranges):
        return None
    ana = Analyzer()
    ranges = []
    for x, y in zip(a.ranges, b.ranges):
        lo = ana.simplify(tir.max(x.min, y.min))
        hi = ana.simplify(tir.min(x.min + x.extent, y.min + y.extent))
        if ana.can_prove(hi <= lo):
            return None
        ranges.append(Range.from_min_extent(lo, ana.simplify(tir.max(0, hi - lo))))
    return Region(a.buffer, ranges, "conservative" if "conservative" in (a.precision, b.precision) else "exact")


def _subtract(region, written):
    overlap = _intersect(region, written)
    if overlap is None:
        return [region]
    if _contains(written, region):
        return []
    ana = Analyzer()
    core = list(region.ranges)
    pieces = []
    for i, (source, cut) in enumerate(zip(region.ranges, overlap.ranges)):
        left = ana.simplify(cut.min - source.min)
        right = ana.simplify(source.min + source.extent - cut.min - cut.extent)
        if not ana.can_prove(left >= 0) or not ana.can_prove(right >= 0):
            return [Region(region.buffer, region.ranges, "conservative")]
        for start, extent in ((source.min, left), (cut.min + cut.extent, right)):
            if not ana.can_prove(extent == 0):
                part = list(core)
                part[i] = Range.from_min_extent(ana.simplify(start), extent)
                pieces.append(Region(region.buffer, part, region.precision))
        core[i] = cut
    return pieces


def _bound(region, domains):
    if not domains:
        return region
    ana = Analyzer()
    ranges = []
    for r in region.ranges:
        low = ana.int_set(r.min, domains)
        high = ana.int_set(r.min + r.extent - 1, domains)
        if low.is_everything() or high.is_everything():
            return Region(region.buffer, region.ranges, "unknown")
        ranges.append(Range.from_min_extent(ana.simplify(low.min_value), ana.simplify(high.max_value - low.min_value + 1)))
    return Region(region.buffer, ranges, "conservative")


def _clip_global(region, loops):
    ana = Analyzer()
    for var, dom, _ in loops:
        ana.bind(var, dom)
    ranges = []
    precision = region.precision
    for r, shape in zip(region.ranges, region.buffer.shape):
        if ana.can_prove(r.min >= 0) and ana.can_prove(r.min + r.extent <= shape):
            ranges.append(r)
        else:
            low = ana.simplify(tir.max(0, r.min))
            high = ana.simplify(tir.min(shape, r.min + r.extent))
            ranges.append(Range.from_min_extent(low, ana.simplify(tir.max(0, high - low))))
            precision = "conservative"
    return Region(region.buffer, ranges, precision)


def _domains(loops):
    return {v: tvm.arith.IntervalSet(r.min, r.min + r.extent - 1) for v, r, _ in loops}


def _store_axes(op, tile_scope=False):
    """Recognize dense separable stores; a bounding box may contain holes."""
    ana = Analyzer()
    axes, used = [], set()
    loop_vars = {v for v, _, kind in op.loops if not tile_scope or kind != "4"}
    for index in op.metadata.indices:
        variables = set()
        tir.stmt_functor.post_order_visit(index, lambda n, variables=variables: variables.add(n) if isinstance(n, tir.Var) else None)
        varying = variables & loop_vars
        if not varying:
            axes.append(None)
            continue
        if len(varying) != 1:
            return None
        var = next(iter(varying))
        if var in used:
            return None
        offset = ana.simplify(index - var)
        remaining = set()
        tir.stmt_functor.post_order_visit(offset, lambda n, remaining=remaining: remaining.add(n) if isinstance(n, tir.Var) else None)
        if remaining & loop_vars:
            return None
        axes.append((var, offset))
        used.add(var)
    return axes


def _map_inputs(op, demand, tile_scope=False):
    meta = op.metadata
    ana = Analyzer()
    if op.kind in ("copy", "async_copy") and len(op.reads) == 1 and len(op.writes) == 1:
        src, dst = op.reads[0], op.writes[0]
        # Tensor slices often retain unit batch/head axes in global memory but
        # copy to a 2D shared tile. Match the non-unit axes without flattening
        # or losing the offsets of those batch/head axes.
        if len(src.ranges) != len(dst.ranges):
            src_axes = [i for i, r in enumerate(src.ranges) if not ana.can_prove_equal(r.extent, 1)]
            dst_axes = [i for i, r in enumerate(dst.ranges) if not ana.can_prove_equal(r.extent, 1)]
            if len(src_axes) == len(dst_axes) and all(
                ana.can_prove_equal(src.ranges[i].extent, dst.ranges[j].extent) for i, j in zip(src_axes, dst_axes)
            ):
                ranges = list(src.ranges)
                for i, j in zip(src_axes, dst_axes):
                    ranges[i] = Range.from_min_extent(
                        ana.simplify(src.ranges[i].min + demand.ranges[j].min - dst.ranges[j].min), demand.ranges[j].extent
                    )
                return [Region(src.buffer, ranges, demand.precision)]
        if len(src.ranges) == len(dst.ranges) and all(ana.can_prove_equal(a.extent, b.extent) for a, b in zip(src.ranges, dst.ranges)):
            return [
                Region(
                    src.buffer,
                    [
                        Range.from_min_extent(ana.simplify(s.min + q.min - d.min), q.extent)
                        for s, d, q in zip(src.ranges, dst.ranges, demand.ranges)
                    ],
                    demand.precision,
                )
            ]
    if meta is not None and hasattr(meta, "aRegion") and hasattr(meta, "transA") and not bool(getattr(meta, "isTcgen05", False)):
        c = Region.from_ir(meta.cRegion)
        if len(demand.ranges) == 2 and len(c.ranges) == 2:
            m, n = [Range.from_min_extent(ana.simplify(q.min - base.min), q.extent) for q, base in zip(demand.ranges, c.ranges)]
            result = []
            for original, axis, sub in ((meta.aRegion, 1 if meta.transA else 0, m), (meta.bRegion, 0 if meta.transB else 1, n)):
                region = Region.from_ir(original)
                if len(region.ranges) != 2:
                    return [Region(r.buffer, r.ranges, "conservative") for r in op.reads]
                base = region.ranges[axis]
                region.ranges[axis] = Range.from_min_extent(ana.simplify(base.min + sub.min), sub.extent)
                result.append(region)
            if _int(meta.clearAccum) != 1:
                result.append(demand)
            return result
    if meta is not None and hasattr(meta, "dim") and hasattr(meta, "srcRegion"):
        src = Region.from_ir(meta.srcRegion)
        dst = Region.from_ir(meta.dstRegion)
        dim = int(meta.dim)
        if len(src.ranges) == len(dst.ranges) + 1:
            j = 0
            for i in range(len(src.ranges)):
                if i != dim:
                    src.ranges[i] = Range.from_min_extent(
                        ana.simplify(src.ranges[i].min + demand.ranges[j].min - dst.ranges[j].min), demand.ranges[j].extent
                    )
                    j += 1
            return [src] + ([demand] if not meta.clear else [])
    if op.kind == "elementwise":
        domains = _domains(tuple(loop for loop in op.loops if not tile_scope or loop[2] != "4"))
        axes = _store_axes(op, tile_scope)
        if axes is not None:
            loop_ranges = {v: r for v, r, _ in op.loops}
            for axis, requested in zip(axes, demand.ranges):
                if axis is not None:
                    var, offset = axis
                    dom = loop_ranges[var]
                    low = ana.simplify(tir.max(dom.min, requested.min - offset))
                    high = ana.simplify(tir.min(dom.min + dom.extent, requested.min + requested.extent - offset) - 1)
                    domains[var] = tvm.arith.IntervalSet(low, high)
        return [_bound(r, domains) for r in op.reads]
    return [Region(r.buffer, r.ranges, "conservative") for r in op.reads]


@dataclass
class PropagationResult:
    operations: list[Operation]
    per_iteration_inputs: list[Region]
    full_loop_inputs: list[Region]
    unknown: list[str]
    input_loops: list = field(default_factory=list)

    def to_dict(self):
        return {
            "operations": [op.to_dict() for op in self.operations],
            "per_iteration_inputs": [r.to_dict() for r in self.per_iteration_inputs],
            "full_loop_inputs": [r.to_dict() for r in self.full_loop_inputs],
            "unknown": self.unknown,
        }


def _propagate(col, outputs, tile_scope=False):
    for op in col.operations:
        op.demands.clear()
    if outputs is None:
        outputs = [
            (_bound(r, _domains(tuple(loop for loop in op.loops if not tile_scope or loop[2] != "4"))) if op.kind == "elementwise" else r)
            for op in col.operations
            for r in op.writes
            if r.buffer.scope() == "global"
        ]
    demands = [r if isinstance(r, Region) else Region.from_ir(r) for r in outputs]
    origins = {id(region): () for region in demands}
    for op in reversed(col.operations):
        matched = []
        for write in op.writes:
            bounded_write = (
                _bound(write, _domains(tuple(loop for loop in op.loops if not tile_scope or loop[2] != "4")))
                if op.kind == "elementwise"
                else write
            )
            for demand in list(demands):
                overlap = _intersect(bounded_write, demand)
                if overlap is not None:
                    op.demands.append(overlap)
                    matched.extend(_map_inputs(op, overlap, tile_scope))
                    # Subtract overwritten rectangles; conditional writes retain
                    # both reaching versions. Unknown overlap keeps a bound.
                    if not op.predicates and (op.kind != "elementwise" or _store_axes(op, tile_scope) is not None):
                        remaining = _subtract(demand, bounded_write)
                        for region in remaining:
                            origins[id(region)] = origins.get(id(demand), ())
                        demands = [r for r in demands if r is not demand] + remaining
        for region in matched:
            demands.append(region)
            origins[id(region)] = () if tile_scope and op.kind == "elementwise" else op.loops
    inputs, coverage, input_loops = [], [], []
    for region in demands:
        if region.buffer.scope() == "global":
            loops = origins.get(id(region), ())
            region = _clip_global(region, loops)
            inputs.append(region)
            input_loops.append(loops)
            coverage.append(_bound(region, _domains(loops)))
    return PropagationResult(col.operations, inputs, coverage, col.unknown, input_loops)


def propagate_inputs(func, outputs=None):
    """Propagate BufferRegion (or Region) output demands through an elaborated PrimFunc.

    Omit outputs to query all global stores. Conservative boxes can overestimate
    traffic; they do not change the kernel's actual allocations or work.
    """
    if not isinstance(func, tir.PrimFunc):
        raise TypeError("propagate_inputs expects an elaborated PrimFunc")
    return _propagate(_Collector(func), outputs)


def analyze_prim_func(func, config=None, outputs=None, *, target=None, device_limits=None, pass_configs=None):
    """Analyze without lowering; resolve register limits from target, function attrs, or target scope."""
    if not isinstance(func, tir.PrimFunc):
        raise TypeError("analyze_prim_func expects an elaborated PrimFunc")
    config = CarverConfig.from_value(config)
    if target is None and func.attrs is not None:
        target = func.attrs.get("target")
    col = _Collector(func)
    propagation = _propagate(col, outputs).to_dict()
    if outputs is not None:
        # Pressure always models the actual kernel, regardless of query size.
        _propagate(col, None)
    pressure = _pressure(col, config, target)
    from .engine import AnalysisContext, run_modules

    context = AnalysisContext(func, col, _propagate(col, None, tile_scope=True), config, pass_configs)
    results = run_modules(context, pressure, device_limits or config.device_limits)
    return {
        **results,
        "propagation": propagation,
        "ir_context": {
            "launch_threads": {k: str(v) for k, v in col.threads.items()},
            "explicit_layouts": {str(k): str(v) for k, v in col.layouts.items()},
        },
    }
