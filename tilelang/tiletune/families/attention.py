"""Online-softmax attention: recognition and policies for the captured graph.

Register estimation uses the shared liveness analysis, retaining the actual
loop-carried output and normalization state. Only this family can opt into a
soft spill allowance. Timing uses both captured GEMMs and every scalar/reduction
operation; actual external accesses prevent duplicate traffic through softmax's
multiple dependency paths. No fixed attention shape or phase cost is supplied.
"""

from ..ir_utils import call_names, dense_gemms, main_loops
from .base import KernelSpecialization, WarpSpecializationPolicy


def _accepts_consumer(op, loop):
    return (
        op.kind in ("gemm", "reduce", "fill", "copy", "elementwise")
        and all(r.buffer.scope() != "global" for r in op.reads + op.writes)
        and not (op.kind == "elementwise" and any(r.buffer.scope().startswith("shared") for r in op.writes))
        and all(kind in ("4", "1") or var.same_as(loop.loop_var) for var, _, kind in op.loops)
    )


ATTENTION_WARP_SPECIALIZATION = WarpSpecializationPolicy(
    consumer_description="attention tile consumers",
    require_tile_calls=False,
    accepts_consumer=_accepts_consumer,
)


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

    def register_spill_allowance(self, config):
        return config.attention_spill_budget_registers_per_thread if self.matched else 0

    def warp_specialization_policy(self):
        return ATTENTION_WARP_SPECIALIZATION

    def memory_traffic(self, context):
        from ..memory import analyze_memory

        # A fused attention graph reuses Q and online state along multiple
        # dependency paths. Charge each actual external tile access once, not
        # once per backward demand path through both GEMMs and softmax.
        return analyze_memory(
            context.collector,
            context.tile_propagation,
            specialization=self,
            actual_accesses=True,
            pass_configs=context.pass_configs,
        )
