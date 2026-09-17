"""Adapter for the SM100 TCGen05 FP8 GEMM example."""

from threading import Lock

from experiments.utils.kernel import KernelCase, _random
from .reference import reference
from .spaces import support_reason

_FP8_GEMM_LOCK = Lock()


def _fp8_gemm_program(**kwargs):
    from examples.gemm_fp8.example_tilelang_gemm_fp8_sm100 import matmul

    with _FP8_GEMM_LOCK:
        return matmul.get_tir(**kwargs)


def make_case(workload):
    reason = support_reason(workload)
    if reason:
        raise ValueError(reason)
    p, dtype = workload.parameters, workload.dtype
    m, n, k = p["m"], p["n"], p["k"]

    def build(block_M, block_N, block_K, num_stages, threads, enable_rasteration):
        func = _fp8_gemm_program(
            M=m,
            N=n,
            K=k,
            block_M=block_M,
            block_N=block_N,
            block_K=block_K,
            trans_A=False,
            trans_B=True,
            in_dtype=dtype,
            out_dtype=dtype,
            accum_dtype="float32",
            num_stages=num_stages,
            threads=threads,
            enable_rasteration=enable_rasteration,
        )
        return func.without_attr("tilelang_out_idx")

    def inputs(device, generator):
        return [_random(shape, dtype, device, generator) for shape in ((m, k), (n, k))]

    return KernelCase(build, inputs, reference, [2], rtol=0.2, atol=0.125)
