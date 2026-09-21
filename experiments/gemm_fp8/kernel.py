"""The original H200 FP8 example, including its rasterization parameter."""

from threading import Lock

from experiments.utils.kernel import KernelCase, _random
from .reference import reference
from .spaces import support_reason

_BUILD_LOCK = Lock()


def make_case(workload):
    reason = support_reason(workload)
    if reason:
        raise ValueError(reason)
    m, n, k = (workload.parameters[key] for key in ("m", "n", "k"))
    dtype = workload.dtype

    def build(block_M, block_N, block_K, num_stages, threads, enable_rasteration):
        from examples.gemm_fp8.example_tilelang_gemm_fp8 import matmul

        with _BUILD_LOCK:
            return matmul.get_tir(
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

    def inputs(device, generator):
        return [_random(shape, dtype, device, generator) for shape in ((m, k), (n, k))]

    return KernelCase(build, inputs, reference, None, rtol=0.03, atol=0.03)
