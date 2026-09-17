"""Explicit block-scaled E4M3 GEMM using the Hopper FP8 example."""

from threading import Lock

from experiments.utils.kernel import KernelCase, _random
from .reference import reference
from .spaces import BLOCK_K, BLOCK_M, NUM_STAGES, THREADS, support_reason

_BUILD_LOCK = Lock()


class BlockScaledFP8KernelCase(KernelCase):
    def check(self, actuals, references):
        if len(actuals) != len(references):
            raise AssertionError("wrong number of kernel outputs")
        if any(actual.shape != expected.shape or actual.dtype != expected.dtype for actual, expected in zip(actuals, references)):
            raise AssertionError("block-scaled FP8 GEMM must return the declared BF16 output")
        super().check(actuals, references)


def _program(**kwargs):
    from examples.deepseek_deepgemm.example_deepgemm_fp8_2xAcc import tl_gemm

    with _BUILD_LOCK:
        return tl_gemm.get_tir(**kwargs)


def make_case(workload):
    reason = support_reason(workload)
    if reason:
        raise ValueError(reason)
    m, n, k = (workload.parameters[key] for key in ("m", "n", "k"))

    def build(block_M, block_N, block_K, num_stages, threads):
        if (block_M, block_K, num_stages, threads) != (BLOCK_M, BLOCK_K, NUM_STAGES, THREADS):
            raise ValueError("Hopper block-scaled FP8 fixes block_M=64, block_K=128, num_stages=4, and threads=128")
        return _program(
            M=m,
            N=n,
            K=k,
            block_N=block_N,
            in_dtype="float8_e4m3fn",
            out_dtype="bfloat16",
            accum_dtype="float32",
        )

    def inputs(device, generator):
        from examples.deepseek_deepgemm.example_deepgemm_fp8_2xAcc import per_block_cast_to_fp8, per_token_cast_to_fp8

        a, scale_a = per_token_cast_to_fp8(_random((m, k), "bfloat16", device, generator))
        b, scale_b = per_block_cast_to_fp8(_random((n, k), "bfloat16", device, generator))
        return [a, b, scale_a, scale_b]

    return BlockScaledFP8KernelCase(build, inputs, reference, None, rtol=0.03, atol=0.03)
