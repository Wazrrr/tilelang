"""Adapter for the repository's real FP8 GEMM example."""

from examples.gemm_fp8.example_tilelang_gemm_fp8 import matmul
from experiments.utils.kernel import KernelCase, _random
from .reference import reference
from .spaces import support_reason


def make_case(workload):
    reason = support_reason(workload)
    if reason:
        raise ValueError(reason)
    p, dtype = workload.parameters, workload.dtype
    m, n, k = p["m"], p["n"], p["k"]

    def build(block_M, block_N, block_K, num_stages, threads, enable_rasteration):
        func = matmul.get_tir(
            M=m,
            N=n,
            K=k,
            block_M=block_M,
            block_N=block_N,
            block_K=block_K,
            dtype=dtype,
            num_stages=num_stages,
            threads=threads,
            enable_rasteration=enable_rasteration,
        )
        return func.without_attr("tilelang_out_idx")

    def inputs(device, generator):
        return [_random(shape, dtype, device, generator) for shape in ((m, k), (n, k))]

    return KernelCase(build, inputs, reference, [2], rtol=0.2, atol=0.125)
