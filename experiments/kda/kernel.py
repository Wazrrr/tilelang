"""Inputs and reference for token-parallel KDA intra-chunk coefficients."""

from threading import Lock

from experiments.utils.kernel import KernelCase, _random
from .reference import chunk_intra_reference


_CHUNK_LOCK = Lock()


def _chunk_program(**kwargs):
    from examples.kda.chunk_intra_token_parallel import tilelang_chunk_kda_fwd_intra_token_parallel

    with _CHUNK_LOCK:
        return tilelang_chunk_kda_fwd_intra_token_parallel.jit_impl.get_tir(**kwargs)


def make_case(w):
    """Build the example's fixed-length BSHD token-parallel intra kernel."""
    p, dtype = w.parameters, w.dtype
    batch, heads, sequence, dim, chunk, sub_chunk = (
        p[key] for key in ("batch", "heads", "sequence", "dim", "chunk_size", "sub_chunk_size")
    )
    shapes = (
        (batch, sequence, heads, dim),
        (batch, sequence, heads, dim),
        (batch, sequence, heads, dim),
        (batch, sequence, heads),
    )

    def build(block_H, num_stages, threads, block_DK=128):
        return _chunk_program(
            B=batch,
            S=sequence,
            H=heads,
            DK=dim,
            input_dtype=dtype,
            output_dtype=dtype,
            accum_dtype="float32",
            gate_dtype="float32",
            chunk_size=chunk,
            sub_chunk_size=sub_chunk,
            scale=dim**-0.5,
            block_H=block_H,
            threads=threads,
            num_stages=num_stages,
            block_DK=block_DK,
        )

    def inputs(device, generator):
        import torch
        import torch.nn.functional as F

        q = _random(shapes[0], dtype, device, generator)
        k = _random(shapes[1], dtype, device, generator)
        gate_logits = _random(shapes[2], "float32", device, generator)
        gates = F.logsigmoid(gate_logits)
        gates = gates.reshape(batch, sequence // chunk, chunk, heads, dim).cumsum(2).reshape(shapes[2])
        beta = _random(shapes[3], dtype, device, generator).sigmoid().to(getattr(torch, dtype))
        return [q, k, gates, beta]

    return KernelCase(
        build,
        inputs,
        chunk_intra_reference(w),
        [4, 5],
        pass_configs={"tl.enable_fast_math": True},
    )
