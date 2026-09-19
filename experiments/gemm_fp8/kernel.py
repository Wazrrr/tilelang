"""Native SM100 E4M3 block-scaled GEMM with TCGen05 and BF16 output."""

from threading import Lock

from experiments.utils.kernel import KernelCase, _random
from .reference import reference
from .spaces import BLOCK_K, BLOCK_M, GROUP_SIZES, NUM_STAGES, STORE_BLOCK_NS, support_reason

_BUILD_LOCK = Lock()


class BlockScaledFP8KernelCase(KernelCase):
    def check(self, actuals, references):
        if len(actuals) != len(references):
            raise AssertionError("wrong number of kernel outputs")
        if any(actual.shape != expected.shape or actual.dtype != expected.dtype for actual, expected in zip(actuals, references)):
            raise AssertionError("block-scaled FP8 GEMM must return the declared BF16 output")
        super().check(actuals, references)


def _program(implementation, **kwargs):
    from examples.blockscaled_gemm_sm100.gemm_mxfp8_blockscaled_1d1d import (
        mxfp8_blockscaled_gemm_2cta,
        mxfp8_blockscaled_gemm_2cta_persistent,
    )

    program = mxfp8_blockscaled_gemm_2cta_persistent if implementation.endswith("persistent") else mxfp8_blockscaled_gemm_2cta
    with _BUILD_LOCK:
        return program.get_tir(**kwargs)


def make_case(workload):
    reason = support_reason(workload)
    if reason:
        raise ValueError(reason)
    m, n, k = (workload.parameters[key] for key in ("m", "n", "k"))

    def build(
        block_M,
        block_N,
        block_K,
        num_stages,
        threads,
        implementation,
        group_size,
        use_tma_store,
        store_block_N,
        column_major=True,
    ):
        if block_M != BLOCK_M or block_N != 256 or block_K != BLOCK_K:
            raise ValueError("SM100 block-scaled FP8 fixes the 128x256x128 two-CTA tile")
        if num_stages not in NUM_STAGES:
            raise ValueError(f"SM100 block-scaled FP8 requires num_stages in {NUM_STAGES}")
        valid = (implementation, threads) in (("tcgen05_2cta", 128), ("tcgen05_2cta_persistent", 256))
        if not valid:
            raise ValueError("invalid native SM100 FP8 implementation/configuration pair")
        kwargs = dict(
            M=m,
            N=n,
            K=k,
            block_M=block_M,
            block_N=block_N,
            block_K=block_K,
            in_dtype="float8_e4m3fn",
            out_dtype="bfloat16",
            accum_dtype="float32",
            num_stages=num_stages,
            sf_granularity_k=128,
            transpose_B=True,
        )
        if implementation.endswith("persistent"):
            if group_size not in GROUP_SIZES or not isinstance(column_major, bool):
                raise ValueError("invalid persistent SM100 tile traversal")
            if use_tma_store and store_block_N not in STORE_BLOCK_NS:
                raise ValueError("invalid persistent SM100 TMA-store tile")
            kwargs.update(
                use_tma_store=use_tma_store,
                store_block_N=store_block_N,
                group_size=group_size,
                column_major=column_major,
            )
        return _program(implementation, **kwargs)

    def inputs(device, generator):
        from examples.blockscaled_gemm_sm100.gemm_mxfp8_blockscaled_1d1d import quantize_fp8_with_packed_ue8m0

        a, scale_a, _ = quantize_fp8_with_packed_ue8m0(_random((m, k), "bfloat16", device, generator), gran_k=128)
        b, scale_b, _ = quantize_fp8_with_packed_ue8m0(_random((n, k), "bfloat16", device, generator), gran_k=128)
        return [a, b, scale_a, scale_b]

    return BlockScaledFP8KernelCase(build, inputs, reference, None, rtol=0.03, atol=0.03)
