"""Kernel graph specializations consumed by the same analysis modules.

Recognition depends on operations and buffer identity, never buffer names,
configuration dictionaries, benchmark results, or reconstructed templates.
"""

from dataclasses import dataclass, field
from tvm import tirx as tir


def effective_pass_configs(context):
    from tilelang.transform import PassContext

    effective = dict(PassContext.current().config)
    effective.update(dict((context.func.attrs or {}).get("tilelang_pass_configs", {})))
    effective.update(context.pass_configs or {})
    return effective


def in_loop(op, loop):
    return loop is not None and any(v.same_as(loop.loop_var) for v, _, _ in op.loops)


def dense_gemms(col):
    return [op for op in col.operations if hasattr(op.metadata, "cRegion") and not bool(getattr(op.metadata, "isTcgen05", False))]


def main_loops(col, operations):
    loops = [loop for loop in col.pipeline_loops if all(in_loop(op, loop) for op in operations)]
    # T.Pipelined(num_stages=0) elaborates to an ordinary serial loop. Recognize
    # its actual copy/compute loop without requiring a disappeared annotation.
    return loops or [loop for loop in col.serial_loops if all(in_loop(op, loop) for op in operations)]


def call_names(op):
    names = []
    if op.kind == "elementwise":
        tir.stmt_functor.post_order_visit(
            op.metadata.value, lambda n: names.append(str(n.op.name)) if isinstance(n, tir.Call) and hasattr(n.op, "name") else None
        )
    return names


@dataclass
class KernelSpecialization:
    name: str = "generic"
    loop: object = None
    roles: dict = field(default_factory=dict)
    evidence: list = field(default_factory=list)
    matched: bool = True

    def register_pressure(self, context, pressure):
        from .register_pressure import analyze_live_tiles

        pressure["tile_liveness"] = analyze_live_tiles(context.collector, self)
        return pressure

    def warp_specialization(self, context, pressure):
        from .warp_specialization import predict_warp_specialization

        return predict_warp_specialization(context.func, context.collector, pressure, context.pass_configs, specialization=self)

    def memory_traffic(self, context):
        from .memory import analyze_memory

        return analyze_memory(
            context.collector, context.tile_propagation, specialization=self, pass_configs=effective_pass_configs(context)
        )

    def pipeline_overlap(self, context, memory, pressure):
        from .pipeline import analyze_pipeline

        effective = effective_pass_configs(context)
        return analyze_pipeline(context.collector, self, memory, pressure, context.config.performance_model, effective)

    def phase(self, op):
        return "mainloop" if in_loop(op, self.loop) else "outside_mainloop"

    def to_dict(self):
        return {
            "name": self.name,
            "matched": self.matched,
            "loop": str(self.loop.loop_var) if self.loop is not None else None,
            "roles": {
                name: {"buffer": b.name, "buffer_id": str(hash(b)), "shape": [str(s) for s in b.shape], "dtype": str(b.dtype)}
                for name, b in self.roles.items()
            },
            "evidence": self.evidence,
        }


class GemmSpecialization(KernelSpecialization):
    @classmethod
    def match(cls, col):
        gemms = dense_gemms(col)
        if len(gemms) != 1:
            return None
        op = gemms[0]
        loops = main_loops(col, [op])
        return cls(
            "gemm",
            loops[0] if len(loops) == 1 else None,
            {"a": op.metadata.a, "b": op.metadata.b, "accumulator": op.metadata.c},
            ["one dense GEMM operation; operand roles taken from reflected metadata"],
        )

    def phase(self, op):
        if hasattr(op.metadata, "cRegion"):
            return "gemm"
        return super().phase(op)


class AttentionSpecialization(KernelSpecialization):
    @classmethod
    def match(cls, col):
        gemms = dense_gemms(col)
        if len(gemms) != 2:
            return None
        first, second = gemms
        loops = main_loops(col, [first, second])
        if len(loops) != 1:
            return None
        # The score tile must feed the probability operand through reductions
        # and an exponential, not merely coexist with another unrelated GEMM.
        derived = {first.metadata.c}
        reductions = set()
        has_exp = False
        for op in col.operations[first.index + 1 : second.index]:
            if any(r.buffer in derived for r in op.reads):
                derived.update(r.buffer for r in op.writes)
                if hasattr(op.metadata, "dim") and hasattr(op.metadata, "srcRegion"):
                    reductions.add(int(op.metadata.type.type))
                has_exp |= any(name in ("tirx.exp", "tirx.exp2") for name in call_names(op))
        if second.metadata.a not in derived or not has_exp or not {0, 2} <= reductions:
            return None
        result = cls(
            "attention",
            loops[0],
            {
                "query": first.metadata.a,
                "key": first.metadata.b,
                "scores": first.metadata.c,
                "probabilities": second.metadata.a,
                "value": second.metadata.b,
                "output_accumulator": second.metadata.c,
            },
            [
                "connected QK GEMM -> max/exp/sum softmax -> PV GEMM inside one tile loop",
                "output accumulator and online normalization state may live across loop iterations",
            ],
        )
        result.score_index, result.value_index = first.index, second.index
        return result

    def phase(self, op):
        if op.index == self.score_index:
            return "qk_gemm"
        if op.index == self.value_index:
            return "pv_gemm"
        if self.score_index < op.index < self.value_index:
            return "softmax_and_rescale"
        return super().phase(op)

    def memory_traffic(self, context):
        from .memory import analyze_memory

        # A fused attention graph reuses Q and online state along multiple
        # dependency paths. Charge each actual external tile access once, not
        # once per backward demand path through both GEMMs and softmax.
        return analyze_memory(
            context.collector,
            context.tile_propagation,
            specialization=self,
            actual_accesses=True,
            pass_configs=effective_pass_configs(context),
        )


def select_specialization(col, requested="auto"):
    for implementation in (AttentionSpecialization, GemmSpecialization):
        result = implementation.match(col)
        if result is not None and requested in ("auto", result.name):
            return result
    return KernelSpecialization(
        matched=requested in ("auto", "generic"),
        evidence=[
            "generic operation analysis"
            if requested in ("auto", "generic")
            else f"requested {requested} specialization did not match the actual operation graph"
        ],
    )
