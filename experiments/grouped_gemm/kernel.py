"""Adapt the concatenated forward example without changing its TileLang program."""

from itertools import accumulate
from threading import Lock

import torch

from experiments.utils.kernel import KernelCase, _random
from .reference import reference
from .spaces import BLOCK_M, support_reason

_GROUPED_GEMM_LOCK = Lock()


def _grouped_program(**kwargs):
    from examples.grouped_gemm.example_grouped_gemm_fwd import grouped_gemm

    # Eager elaboration mutates the builder; compilation remains parallel.
    with _GROUPED_GEMM_LOCK:
        return grouped_gemm.get_tir(**kwargs)


def make_case(workload):
    reason = support_reason(workload)
    if reason:
        raise ValueError(reason)
    p, dtype = workload.parameters, workload.dtype
    sizes = tuple(p["batch_sizes"])
    sizes_csv = ",".join(map(str, sizes))
    n, k, transpose_b = p["n"], p["k"], p.get("transpose_b", False)

    def build(block_M, block_N, block_K, num_stages, threads):
        if block_M != BLOCK_M:
            raise ValueError(f"grouped GEMM inputs require block_M={BLOCK_M}")
        return _grouped_program(
            K=k,
            N=n,
            batch_sizes_list=tuple(map(int, sizes_csv.split(","))),
            trans_b=transpose_b,
            dtype=dtype,
            block_M=block_M,
            block_N=block_N,
            block_K=block_K,
            num_stages=num_stages,
            threads=threads,
        )

    def inputs(device, generator):
        offsets = list(accumulate(sizes, initial=0))[:-1]
        padded = [((size + BLOCK_M - 1) // BLOCK_M) * BLOCK_M for size in sizes]
        padded_offsets = list(accumulate(padded, initial=0))[:-1]
        b_shape = (len(sizes), n, k) if transpose_b else (len(sizes), k, n)
        return [
            _random((sum(sizes), k), dtype, device, generator),
            _random(b_shape, dtype, device, generator),
            *(torch.tensor(values, dtype=torch.int32, device=device) for values in (sizes, offsets, padded_offsets)),
        ]

    offsets = list(accumulate(sizes, initial=0))[:-1]
    padded = [((size + BLOCK_M - 1) // BLOCK_M) * BLOCK_M for size in sizes]
    return KernelCase(
        build,
        inputs,
        lambda a, b, *metadata: reference(a, b, sizes, transpose_b),
        None,
        rtol=0.01,
        atol=0.01,
        input_values={"2": list(sizes), "3": offsets, "4": list(accumulate(padded, initial=0))[:-1]},
    )
