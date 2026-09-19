"""One block-scale contract, with exact BF16 operand conversion on Ampere."""

from threading import Lock

from experiments.backend import FP8_COMPUTE_DTYPE
from experiments.utils.kernel import KernelCase, _random
from .reference import reference
from .spaces import BLOCK_K, support_reason

_BUILD_LOCK = Lock()


def make_case(workload):
    reason = support_reason(workload)
    if reason:
        raise ValueError(reason)
    m, n, k = (workload.parameters[key] for key in ("m", "n", "k"))

    def build(block_M, block_N, block_K, num_stages, threads):
        from examples.gemm_fp8.example_blockscaled_gemm import blockscaled_gemm

        if block_K != BLOCK_K:
            raise ValueError("the common FP8 scale layout fixes block_K=128")
        with _BUILD_LOCK:
            return blockscaled_gemm.get_tir(
                M=m,
                N=n,
                K=k,
                block_M=block_M,
                block_N=block_N,
                num_stages=num_stages,
                threads=threads,
                compute_dtype=FP8_COMPUTE_DTYPE,
            )

    def inputs(device, generator):
        from examples.gemm_fp8.example_blockscaled_gemm import quantize_e4m3

        a, scale_a = quantize_e4m3(_random((m, k), "bfloat16", device, generator))
        b, scale_b = quantize_e4m3(_random((n, k), "bfloat16", device, generator))
        return [a, b, scale_a, scale_b]

    return KernelCase(build, inputs, reference, None, rtol=0.03, atol=0.03)
