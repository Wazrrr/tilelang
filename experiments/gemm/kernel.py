"""Inputs and numerical reference for the SM100 TCGen05 GEMM example."""

from threading import Lock

from experiments.utils.kernel import KernelCase, _random
from .reference import reference
from .spaces import support_reason

_GEMM_LOCK = Lock()


def _gemm_program(**kwargs):
    from examples.gemm_sm100.gemm_tcgen5mma import matmul

    with _GEMM_LOCK:
        return matmul.get_tir(**kwargs)


def make_case(workload):
    reason = support_reason(workload)
    if reason:
        raise ValueError(reason)
    p, dtype = workload.parameters, workload.dtype
    m, n, k = p["m"], p["n"], p["k"]

    def build(block_M, block_N, block_K, num_stages, thread_num, enable_rasteration):
        return _gemm_program(
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
            threads=thread_num,
            enable_rasteration=enable_rasteration,
        ).without_attr("tilelang_out_idx")

    def inputs(device, generator):
        return [_random(shape, dtype, device, generator) for shape in ((m, k), (n, k))]

    return KernelCase(build, inputs, reference, [2])
