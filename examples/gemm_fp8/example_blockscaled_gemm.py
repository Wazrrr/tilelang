"""E4M3 GEMM with fixed per-row, per-128-K FP32 scales on both operands.

Ampere converts E4M3 values exactly to BF16 before tensor-core compute.
Hopper and Blackwell use native E4M3 operands. Both paths scale each FP32
partial product and return BF16, with the same public layout.
"""

import torch
import tilelang
import tilelang.language as T


@tilelang.jit
def blockscaled_gemm(A, B, scale_a, scale_b, block_M, block_N, num_stages, threads, compute_dtype):
    M, N, K = T.const("M, N, K")
    block_K = 128
    A: T.Tensor((M, K), T.float8_e4m3fn)
    B: T.Tensor((N, K), T.float8_e4m3fn)
    scale_a: T.Tensor((M, T.ceildiv(K, block_K)), T.float32)
    scale_b: T.Tensor((N, T.ceildiv(K, block_K)), T.float32)
    C = T.empty((M, N), T.bfloat16)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), compute_dtype)
        B_shared = T.alloc_shared((block_N, block_K), compute_dtype)
        scale_a_shared = T.alloc_shared((block_M, 1), T.float32)
        scale_b_shared = T.alloc_shared((block_N, 1), T.float32)
        partial = T.alloc_fragment((block_M, block_N), T.float32)
        accum = T.alloc_fragment((block_M, block_N), T.float32)
        C_shared = T.alloc_shared((block_M, block_N), T.bfloat16)
        T.clear(accum)
        for kb in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
            T.copy(A[by * block_M, kb * block_K], A_shared)
            T.copy(B[bx * block_N, kb * block_K], B_shared)
            T.copy(scale_a[by * block_M : (by + 1) * block_M, kb : kb + 1], scale_a_shared)
            T.copy(scale_b[bx * block_N : (bx + 1) * block_N, kb : kb + 1], scale_b_shared)
            T.gemm(A_shared, B_shared, partial, transpose_B=True, clear_accum=True)
            for i, j in T.Parallel(block_M, block_N):
                accum[i, j] += partial[i, j] * scale_a_shared[i, 0] * scale_b_shared[j, 0]
        T.copy(accum, C_shared)
        T.copy(C_shared, C[by * block_M, bx * block_N])
    return C


def quantize_e4m3(x):
    """The scale grid is independent of the chosen kernel tile."""
    if x.ndim != 2 or x.shape[1] % 128:
        raise ValueError("block-scaled operands require a matrix with K divisible by 128")
    rows, k = x.shape
    blocks = x.float().reshape(rows, k // 128, 128)
    scale = blocks.abs().amax(dim=2).clamp_min(1e-4) / 448.0
    values = (blocks / scale.unsqueeze(2)).to(torch.float8_e4m3fn).reshape(rows, k)
    return values, scale
