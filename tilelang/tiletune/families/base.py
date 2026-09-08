"""Family contract and shared implementations of the analysis stages.

The engine calls these stages for every family. Subclasses describe the actual
graph and supply policy differences; liveness, service-time equations and buffer
scheduling remain common. The generic fallback uses the same conservative dense
consumer restrictions as GEMM when predicting warp specialization.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from ..ir_utils import in_loop


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

    def register_pressure(self, context, pressure):
        """Estimate live storage from actual reads/writes, including loop state.

        The proven accumulator lower bound is already in ``pressure``. This
        conservative liveness estimate does not strengthen that proof.
        """
        from ..liveness import analyze_live_tiles

        pressure["tile_liveness"] = analyze_live_tiles(context.collector, self)
        return pressure

    def register_spill_allowance(self, config):
        """Soft demand margin; physical register capacity is never enlarged."""
        return 0

    def warp_specialization_policy(self):
        from .gemm import GEMM_WARP_SPECIALIZATION

        return GEMM_WARP_SPECIALIZATION

    def warp_specialization(self, context, pressure):
        """Combine family consumer restrictions with the shared Hopper policy."""
        from ..warp_specialization import predict_warp_specialization

        return predict_warp_specialization(context.func, context.collector, pressure, context.pass_configs, specialization=self)

    def memory_traffic(self, context):
        """Default accounting uses backward-propagated external input tiles."""
        from ..memory import analyze_memory

        return analyze_memory(context.collector, context.tile_propagation, specialization=self, pass_configs=context.pass_configs)

    def pipeline_overlap(self, context, memory, pressure):
        """Time captured operations in program order using the shared model.

        ``phase`` provides family labels. Work, copies, buffer lifetimes, loop
        bounds and stage depth come from the collector, not a family template.
        """
        from ..pipeline import analyze_pipeline

        return analyze_pipeline(context.collector, self, memory, pressure, context.config.performance_model, context.pass_configs)

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
