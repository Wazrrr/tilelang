"""E4M3 storage with explicit scales and BF16 tensor-core compute on Ampere."""

from threading import Lock

from experiments.utils.kernel import KernelCase, _random
from .reference import reference
from .spaces import BLOCK_K, support_reason

_BUILD_LOCK = Lock()


class EmulatedFP8KernelCase(KernelCase):
    def check(self, actuals, references):
        if len(actuals) != len(references):
            raise AssertionError("wrong number of kernel outputs")
        if any(actual.shape != expected.shape or actual.dtype != expected.dtype for actual, expected in zip(actuals, references)):
            raise AssertionError("Ampere block-scaled GEMM must return the declared BF16 output")
        super().check(actuals, references)


def _program(**kwargs):
    from examples.gemm_fp8.example_mxfp8_blockscaled_gemm_a100 import blockscaled_gemm

    with _BUILD_LOCK:
        return blockscaled_gemm.get_tir(**kwargs)


def make_case(workload):
    reason = support_reason(workload)
    if reason:
        raise ValueError(reason)
    m, n, k = (workload.parameters[key] for key in ("m", "n", "k"))

    def build(block_M, block_N, block_K, num_stages, threads):
        if block_K != BLOCK_K:
            raise ValueError("the explicit Ampere scale layout fixes block_K=128")
        return _program(
            M=m,
            N=n,
            K=k,
            block_M=block_M,
            block_N=block_N,
            num_stages=num_stages,
            threads=threads,
        )

    def inputs(device, generator):
        from examples.gemm_fp8.example_mxfp8_blockscaled_gemm_a100 import quantize_e4m3_1d1d

        a, scale_a = quantize_e4m3_1d1d(_random((m, k), "bfloat16", device, generator))
        b, scale_b = quantize_e4m3_1d1d(_random((n, k), "bfloat16", device, generator))
        return [a, b, scale_a, scale_b]

    return EmulatedFP8KernelCase(build, inputs, reference, None, rtol=0.03, atol=0.03)
