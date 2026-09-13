"""Elaborate the existing FP8 GEMM safely for parallel compilation workers."""

from threading import Lock

import torch

from examples.gemm_fp8.example_gemm_fp8_tiletune import get_configs as get_configs
from examples.gemm_fp8.example_tilelang_gemm_fp8 import calc_diff, matmul


_ELABORATION_LOCK = Lock()


def make_kernel(M, N, K, dtype):
    def gemm(block_M, block_N, block_K, num_stages, threads, enable_rasteration):
        # get_tir mutates the eager example's AST builder. Only construction is
        # serialized; lowering, grouped compilation, and benchmarking stay parallel.
        with _ELABORATION_LOCK:
            program = matmul.get_tir(
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
        # Both compilation paths receive out_idx=[2] from the experiment runner.
        # Remove eager-JIT return metadata so standalone compilation agrees with
        # grouped compilation, which requires explicit output indices.
        return program.without_attr("tilelang_out_idx")

    return gemm


def make_inputs(M, N, K, dtype, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return [
        torch.empty(shape, device="cuda", dtype=torch.float16).uniform_(-0.5, 0.5, generator=generator).to(getattr(torch, dtype))
        for shape in ((M, K), (N, K))
    ]


def reference(a, b):
    return (a.float() @ b.float().T).to(a.dtype)


def check_accuracy(actuals, refs):
    difference = calc_diff(actuals[0], refs[0]).item()
    if not difference < 1e-3:
        raise AssertionError(f"FP8 calc_diff={difference}; expected < 1e-3")
