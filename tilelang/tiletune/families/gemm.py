"""Dense GEMM: recognition, register policy and pipeline consumer policy.

Register estimation inherits the common accumulator bound and live-tile model,
with no soft spill allowance. Timing inherits the common per-operation service
and buffer schedule, using propagated input tiles for traffic. Positive-stage
warp specialization currently supports straight-line copies and GEMM consumers.
"""

from ..src.ir_utils import dense_gemms, main_loops
from .base import KernelSpecialization, WarpSpecializationPolicy


def _accepts_consumer(op, loop):
    return hasattr(op.metadata, "cRegion")


GEMM_WARP_SPECIALIZATION = WarpSpecializationPolicy(
    consumer_description="dense GEMM consumers",
    require_tile_calls=True,
    accepts_consumer=_accepts_consumer,
)


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
