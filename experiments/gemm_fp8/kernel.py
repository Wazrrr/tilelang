"""Elaborate the existing FP8 GEMM safely for parallel compilation workers."""

from threading import Lock

from examples.gemm_fp8.example_gemm_fp8_new_carver import get_configs as get_configs
from examples.gemm_fp8.example_tilelang_gemm_fp8 import matmul


_ELABORATION_LOCK = Lock()


def make_kernel(M, N, K, dtype):
    def gemm(block_M, block_N, block_K, num_stages, threads, enable_rasteration):
        # get_tir mutates the eager example's AST builder. Only construction is
        # serialized; lowering, grouped compilation, and benchmarking stay parallel.
        with _ELABORATION_LOCK:
            return matmul.get_tir(
                M=M,
                N=N,
                K=K,
                dtype=dtype,
                block_M=block_M,
                block_N=block_N,
                block_K=block_K,
                num_stages=num_stages,
                threads=threads,
                enable_rasteration=enable_rasteration,
            )

    return gemm
