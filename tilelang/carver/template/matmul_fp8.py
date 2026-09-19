"""Block-scaled E4M3 semantics for the shared experiment contract."""

from dataclasses import dataclass, field

from tvm import te

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
    """Block-scaled FP8 inputs with BF16 output and FP32 accumulation."""

    in_dtype: str = field(default="float8_e4m3", init=False)
    out_dtype: str = field(default="bfloat16", init=False)
    accum_dtype: str = field(default="float32", init=False)
    with_bias: bool = field(default=False, init=False)
    kernel_dtype: str = "float8_e4m3fn"
    compute_dtype: str = "float8_e4m3"

    def initialize_function(self) -> None:
        if self.kernel_dtype not in _MODEL_DTYPES:
            raise ValueError("FP8 GEMM requires float8_e4m3fn, float8_e4m3, or float8_e5m2")
        self.in_dtype = _MODEL_DTYPES[self.kernel_dtype]
        if self.K % 128:
            raise ValueError("the FP8 scale layout requires K divisible by 128")
        if self.trans_A or not self.trans_B or self.with_bias:
            raise ValueError("block-scaled FP8 requires A=(M,K), B=(N,K), without bias")
        blocks = self.K // 128
        a = te.placeholder((self.M, self.K), self.in_dtype, "A")
        b = te.placeholder((self.N, self.K), self.in_dtype, "B")
        sa = te.placeholder((self.M, blocks), "float32", "ScaleA")
        sb = te.placeholder((self.N, blocks), "float32", "ScaleB")
        rk = te.reduce_axis((0, 128), "k")
        compute_dtype = _MODEL_DTYPES.get(self.compute_dtype, self.compute_dtype)
        partial = te.compute(
            (blocks, self.M, self.N),
            lambda block, i, j: te.sum(
                a[i, block * 128 + rk].astype(compute_dtype).astype("float32")
                * b[j, block * 128 + rk].astype(compute_dtype).astype("float32"),
                rk,
            ),
            name="Partial",
        )
        scaled = te.compute(partial.shape, lambda block, i, j: partial[block, i, j] * sa[i, block] * sb[j, block], name="Scaled")
        rb = te.reduce_axis((0, blocks), "block")
        accum = te.compute((self.M, self.N), lambda i, j: te.sum(scaled[rb, i, j], rb), name="Accum")
        output = te.compute(accum.shape, lambda i, j: accum[i, j].astype(self.out_dtype), name="Output")
        self.set_function(te.create_prim_func([a, b, sa, sb, output]))

    def params_as_dict(self):
        return {**super().params_as_dict(), "kernel_dtype": self.kernel_dtype, "model_dtype": self.in_dtype}
