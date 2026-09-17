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


@dataclass
class GroupedMXFP8MatmulTemplate(GroupedMatmulTemplate):
    """Padded grouped MXFP8 GEMM with block scales and BF16 output.

    The equivalent dense graph represents the arithmetic domain seen by
    Carver. The experiment adapter separately models scale traffic and the
    native SM100 kernel's two-CTA cluster dispatch.
    """

    in_dtype: str = field(default="float8_e4m3", init=False)
    out_dtype: str = field(default="bfloat16", init=False)
    accum_dtype: str = field(default="float32", init=False)
    with_bias: bool = field(default=False, init=False)
    kernel_dtype: str = "float8_e4m3fn"
    scale_granularity_k: int = 128
    cluster_size: int = 2

    def initialize_function(self) -> None:
        if self.kernel_dtype != "float8_e4m3fn":
            raise ValueError("grouped MXFP8 GEMM requires float8_e4m3fn inputs")
        if self.scale_granularity_k <= 0 or self.K % self.scale_granularity_k:
            raise ValueError("K must be divisible by the MXFP8 scale granularity")
        if self.cluster_size != 2:
            raise ValueError("the native grouped MXFP8 kernel requires two-CTA clusters")
        super().initialize_function()

    def params_as_dict(self):
        return {
            **super().params_as_dict(),
            "kernel_dtype": self.kernel_dtype,
            "scale_granularity_k": self.scale_granularity_k,
            "cluster_size": self.cluster_size,
        }
