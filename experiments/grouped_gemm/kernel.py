"""Adapt the SM100 grouped MXFP8 block-scaled GEMM example."""

from itertools import accumulate
from threading import Lock

import torch

from experiments.utils.kernel import KernelCase, _random
from .reference import reference
from .spaces import BLOCK_K, BLOCK_M, BLOCK_N, NUM_STAGES, support_reason

_GROUPED_GEMM_LOCK = Lock()


def _grouped_program(persistent, **kwargs):
    from examples.blockscaled_gemm_sm100.grouped_gemm_mxfp8_blockscaled_1d1d import (
        grouped_mxfp8_blockscaled_gemm_2cta,
        grouped_mxfp8_blockscaled_gemm_2cta_persistent,
    )

    program = grouped_mxfp8_blockscaled_gemm_2cta_persistent if persistent else grouped_mxfp8_blockscaled_gemm_2cta
    with _GROUPED_GEMM_LOCK:
        return program.get_tir(**kwargs)


def make_case(workload):
    reason = support_reason(workload)
    if reason:
        raise ValueError(reason)
    p, dtype = workload.parameters, workload.dtype
    sizes = tuple(p["batch_sizes"])
    n, k, transpose_b = p["n"], p["k"], p.get("transpose_b", False)
    m_total, experts, max_m = sum(sizes), len(sizes), max(sizes)
    storage_sizes = tuple(((size + BLOCK_M - 1) // BLOCK_M) * BLOCK_M for size in sizes)
    m_storage = sum(storage_sizes)

    def build(block_M, block_N, block_K, num_stages, threads, persistent):
        expected_threads = 256 if persistent else 128
        expected = (BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, expected_threads)
        if (block_M, block_N, block_K, num_stages, threads) != expected:
            raise ValueError(
                "grouped MXFP8 requires block_M=128, block_N=256, block_K=128, "
                f"num_stages=6 and threads={expected_threads} when persistent={persistent}"
            )
        return _grouped_program(
            persistent=persistent,
            M_storage=m_storage,
            N=n,
            K=k,
            E=experts,
            E1=experts + 1,
            logical_M_total=m_total,
            block_M=block_M,
            block_N=block_N,
            block_K=block_K,
            in_dtype=dtype,
            out_dtype="bfloat16",
            accum_dtype="float32",
            num_stages=num_stages,
            max_M_per_E=max_m,
            transpose_B=transpose_b,
            sf_granularity_k=128,
        )

    def inputs(device, generator):
        from examples.blockscaled_gemm_sm100.grouped_gemm_mxfp8_blockscaled_1d1d import (
            pack_rows_to_group_major_flat,
            pack_sfb_to_group_major_flat,
            quantize_fp8_with_packed_ue8m0_rows,
        )

        a_source = torch.zeros((m_storage, k), dtype=torch.float16, device=device)
        logical_a = _random((m_total, k), "float16", device, generator)
        logical_start = 0
        storage_starts = list(accumulate(storage_sizes, initial=0))
        for group, size in enumerate(sizes):
            storage_start = storage_starts[group]
            a_source[storage_start : storage_start + size] = logical_a[logical_start : logical_start + size]
            logical_start += size
        a, sfa, _ = quantize_fp8_with_packed_ue8m0_rows(a_source)
        b_nt, sfb, _ = quantize_fp8_with_packed_ue8m0_rows(
            _random((experts * n, k), "float16", device, generator)
        )
        b_nt = b_nt.view(experts, n, k).contiguous()
        sfb = sfb.view(experts, n, -1).contiguous()
        b = b_nt if transpose_b else b_nt.transpose(1, 2).contiguous()
        offsets = torch.tensor(list(accumulate(sizes, initial=0)), dtype=torch.int32, device=device)
        storage_offsets = torch.tensor(storage_starts, dtype=torch.int32, device=device)
        return [
            a,
            b,
            pack_rows_to_group_major_flat(sfa),
            pack_sfb_to_group_major_flat(sfb),
            offsets,
            storage_offsets,
        ]

    return KernelCase(
        build,
        inputs,
        lambda a, b, sfa, sfb, offsets, storage_offsets: reference(
            a, b, sfa, sfb, offsets, storage_offsets, transpose_b
        ),
        None,
        rtol=0.02,
        atol=0.02,
    )
