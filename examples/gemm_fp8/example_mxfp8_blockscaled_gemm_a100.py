"""Block-scaled E4M3 GEMM for Ampere via BF16 tensor-core emulation.

Ampere has no native FP8 matrix instruction. This kernel keeps E4M3 storage and
explicit per-row, per-128-K-block scales, converts tiles to BF16 in shared
memory, and performs FP32-accumulating BF16 tensor-core GEMMs.
"""

from typing import Tuple

import torch
import tilelang
import tilelang.language as T


@tilelang.jit
def blockscaled_gemm(A, B, scale_a, scale_b, block_M, block_N, num_stages, threads):
    M, N, K = T.const("M, N, K")
    block_K = 128

    A: T.Tensor((M, K), T.float8_e4m3fn)
    B: T.Tensor((N, K), T.float8_e4m3fn)
    scale_a: T.Tensor((M, T.ceildiv(K, block_K)), T.float32)
    scale_b: T.Tensor((N, T.ceildiv(K, block_K)), T.float32)
    C = T.empty((M, N), T.bfloat16)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), T.bfloat16)
        B_shared = T.alloc_shared((block_N, block_K), T.bfloat16)
        C_partial = T.alloc_fragment((block_M, block_N), T.float32)
        C_accum = T.alloc_fragment((block_M, block_N), T.float32)
        C_shared = T.alloc_shared((block_M, block_N), T.bfloat16)

        T.clear(C_accum)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[bx * block_N, k * block_K], B_shared)
            T.clear(C_partial)
            T.gemm(A_shared, B_shared, C_partial, transpose_B=True)
            for i, j in T.Parallel(block_M, block_N):
                C_accum[i, j] += C_partial[i, j] * scale_a[by * block_M + i, k] * scale_b[bx * block_N + j, k]

        T.copy(C_accum, C_shared)
        T.copy(C_shared, C[by * block_M, bx * block_N])
    return C


def quantize_e4m3_1d1d(x: torch.Tensor, granularity_k: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize rows independently for every fixed K block."""
    assert x.dim() == 2 and x.shape[1] % granularity_k == 0
    rows, k = x.shape
    view = x.view(rows, k // granularity_k, granularity_k)
    amax = view.abs().float().amax(dim=2).clamp_min(1e-4)
    scale = amax / 448.0
    quantized = (view / scale.unsqueeze(2)).to(torch.float8_e4m3fn).view(rows, k)
    return quantized, scale
