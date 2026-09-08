"""The advanced-autotune GEMM (A @ B.T), shared by both experiment types."""

import itertools

import tilelang.language as T
import torch


def get_configs():
    keys = ("block_M", "block_N", "block_K", "num_stages", "thread_num", "enable_rasteration")
    values = itertools.product([64, 128, 256], [64, 128, 256], [32, 64], [0, 1, 2, 3], [128, 256], [True, False])
    return [dict(zip(keys, item)) for item in values]


def make_kernel(M, N, K, dtype):
    # Same tiles, shared epilogue, swizzle, and FP32 accumulation as
    # examples/gemm/example_gemm_advanced_autotune.py::make_autotune_kernel_builder.
    def gemm(block_M, block_N, block_K, num_stages, thread_num, enable_rasteration):
        @T.prim_func
        def main(A: T.Tensor((M, K), dtype), B: T.Tensor((N, K), dtype), C: T.Tensor((M, N), dtype)):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
                a_shared = T.alloc_shared((block_M, block_K), dtype)
                b_shared = T.alloc_shared((block_N, block_K), dtype)
                c_local = T.alloc_fragment((block_M, block_N), "float32")
                c_shared = T.alloc_shared((block_M, block_N), dtype)
                T.use_swizzle(panel_size=10, enable=enable_rasteration)
                T.clear(c_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], a_shared)
                    T.copy(B[bx * block_N, k * block_K], b_shared)
                    T.gemm(a_shared, b_shared, c_local, transpose_B=True)
                T.copy(c_local, c_shared)
                T.copy(c_shared, C[by * block_M, bx * block_N])

        return main

    return gemm


def make_inputs(M, N, K, dtype, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return [
        torch.empty(shape, device="cuda", dtype=getattr(torch, dtype)).uniform_(-0.5, 0.5, generator=generator)
        for shape in ((M, K), (N, K))
    ]


def reference(a, b):
    return a @ b.T
