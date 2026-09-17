"""Carver template for the padded-CTA grouped GEMM used by TileLang."""

from dataclasses import dataclass, field

from .matmul import MatmulTemplate


@dataclass
class GroupedMatmulTemplate(MatmulTemplate):
    """Equivalent dense domain for a grouped kernel with fixed M-sized CTAs.

    The example dispatches one full M tile for every ``ceil(group_m / block_m)``
    unit. Flattening those padded units preserves its CTA count, tensor-core
    work, and tile storage while group offsets affect addresses only.
    """

    batch_sizes: list[int] = field(default_factory=list)
    block_m: int = 64

    def initialize_function(self) -> None:
        if not self.batch_sizes or any(type(size) is not int or size <= 0 for size in self.batch_sizes):
            raise ValueError("batch_sizes must contain positive integers")
        if self.block_m <= 0:
            raise ValueError("block_m must be positive")
        self.M = sum((size + self.block_m - 1) // self.block_m for size in self.batch_sizes) * self.block_m
        super().initialize_function()

    def params_as_dict(self):
        return {**super().params_as_dict(), "batch_sizes": self.batch_sizes, "block_m": self.block_m}
