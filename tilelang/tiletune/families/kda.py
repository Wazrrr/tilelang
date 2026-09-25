"""Chunk output with a gated query/state loop and a final local GEMM."""

from ..src.ir_utils import call_names, dense_gemms, in_loop, main_loops
from .base import KernelSpecialization, WarpSpecializationPolicy


def _accepts_consumer(op, loop):
    return (
        op.kind in ("gemm", "elementwise")
        and all(r.buffer.scope() != "global" for r in op.reads + op.writes)
        and all(kind in ("4", "1") or var.same_as(loop.loop_var) for var, _, kind in op.loops)
    )


class KDAChunkOutputSpecialization(KernelSpecialization):
    @classmethod
    def match(cls, col):
        gemms = dense_gemms(col)
        if len(gemms) != 2:
            return None
        carried, local = gemms
        loops = main_loops(col, [carried])
        if len(loops) != 1 or in_loop(local, loops[0]) or not carried.metadata.c.same_as(local.metadata.c):
            return None
        gates = [
            op
            for op in col.operations[: carried.index]
            if in_loop(op, loops[0]) and any(r.buffer.same_as(carried.metadata.a) for r in op.writes) and "tirx.exp2" in call_names(op)
        ]
        if len(gates) != 1 or not carried.metadata.a.scope().startswith("shared"):
            return None
        result = cls(
            "kda_chunk_o",
            loops[0],
            {"gated_query": carried.metadata.a, "state": carried.metadata.b, "output_accumulator": carried.metadata.c},
            ["shared query gating and query/state GEMM in one loop; a second GEMM updates the same accumulator outside it"],
        )
        result.carried_index, result.local_index = carried.index, local.index
        return result

    @property
    def actual_memory_accesses(self):
        return True

    def warp_specialization_policy(self):
        return WarpSpecializationPolicy("shared query gating and GEMM consumers", False, _accepts_consumer)

    def phase(self, op):
        if op.index == self.carried_index:
            return "query_state_gemm"
        if op.index == self.local_index:
            return "attention_value_gemm"
        return super().phase(op)
