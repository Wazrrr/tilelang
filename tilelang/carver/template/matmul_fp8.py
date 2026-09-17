"""Canonical Carver template for the repository's native FP8 GEMM kernel."""

from dataclasses import dataclass, field

from .matmul import MatmulTemplate


_MODEL_DTYPES = {
    # TileLang/PyTorch spell the finite-only format e4m3fn. The legacy Carver
    # tensorization vocabulary calls the same CUDA tensor-core operand e4m3.
    "float8_e4m3fn": "float8_e4m3",
    "float8_e4m3": "float8_e4m3",
    "float8_e5m2": "float8_e5m2",
}


@dataclass
class FP8MatmulTemplate(MatmulTemplate):
    """FP8 inputs/output with FP32 accumulation and explicit dtype lowering."""

    in_dtype: str = field(default="float8_e4m3", init=False)
    out_dtype: str = field(default="float8_e4m3", init=False)
    accum_dtype: str = field(default="float32", init=False)
    with_bias: bool = field(default=False, init=False)
    kernel_dtype: str = "float8_e4m3fn"

    def initialize_function(self) -> None:
        if self.kernel_dtype not in _MODEL_DTYPES:
            raise ValueError("FP8 GEMM requires float8_e4m3fn, float8_e4m3, or float8_e5m2")
        self.in_dtype = _MODEL_DTYPES[self.kernel_dtype]
        self.out_dtype = self.in_dtype
        super().initialize_function()

    def params_as_dict(self):
        return {**super().params_as_dict(), "kernel_dtype": self.kernel_dtype, "model_dtype": self.in_dtype}
