"""Family recognition results and analysis policy inputs.

Subclasses describe the actual graph and supply policy differences. The engine
calls shared analysis stages directly. The generic fallback uses the same dense
consumer restrictions as GEMM when predicting warp specialization.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from ..src.ir_utils import in_loop


@dataclass(frozen=True)
class WarpSpecializationPolicy:
    """Family restrictions applied before the common native producer checks."""

    consumer_description: str
    require_tile_calls: bool
    accepts_consumer: Callable[[object, object], bool]


@dataclass
class KernelSpecialization:
    name: str = "generic"
    loop: object = None
    roles: dict = field(default_factory=dict)
    evidence: list = field(default_factory=list)
    matched: bool = True

    @property
    def actual_memory_accesses(self):
        return False

    def register_spill_allowance(self, config):
        """Soft demand margin; physical register capacity is never enlarged."""
        return 0

    def warp_specialization_policy(self):
        from .gemm import GEMM_WARP_SPECIALIZATION

        return GEMM_WARP_SPECIALIZATION

    def phase(self, op):
        """Label the same operation for register-liveness and timing reports."""
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
