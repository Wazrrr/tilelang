"""Independent FP32 reference for SM100's packed UE8M0 scale layout."""


def reference(a, b, scale_a, scale_b):
    from examples.blockscaled_gemm_sm100.gemm_mxfp8_blockscaled_1d1d import blockscaled_gemm_ref

    return blockscaled_gemm_ref(a, b, scale_a, scale_b, sf_granularity_k=128, transpose_B=True).bfloat16()
