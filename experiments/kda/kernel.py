"""Inputs and reference for the example KDA chunk-output kernel."""

from threading import Lock
from experiments.utils.kernel import KernelCase, _random
from .reference import chunk_reference


_CHUNK_LOCK = Lock()


def _chunk_program(**kwargs):
    from examples.kda.chunk_o import tilelang_chunk_fwd_o

    with _CHUNK_LOCK:
        return tilelang_chunk_fwd_o.jit_impl.get_tir(**kwargs)


def _gate(shape, chunk, device, generator):
    """FP32 base-2 cumulative log gates, reset at each chunk boundary."""
    import torch

    b, s, h, d = shape
    increments = -(torch.rand(shape, device=device, generator=generator) + 0.1) / chunk
    return increments.reshape(b, s // chunk, chunk, h, d).cumsum(2).reshape(shape)


def make_case(w):
    """The example's BSHD chunk-output kernel, without a rewritten algorithm."""
    p, dtype = w.parameters, w.dtype
    batch, heads, sequence, dk, dv, chunk = (p[k] for k in ("batch", "heads", "sequence", "dim", "value_dim", "chunk_size"))
    shapes = (
        (batch, sequence, heads, dk),
        (batch, sequence, heads, dv),
        (batch, sequence, heads, dk),
        (batch, sequence, heads, chunk),
        (batch, sequence // chunk, heads, dk, dv),
    )

    def build(block_DK, block_DV, num_stages, threads):
        return _chunk_program(
            B=batch,
            S=sequence,
            H=heads,
            DK=dk,
            DV=dv,
            input_dtype=dtype,
            output_dtype=dtype,
            accum_dtype="float32",
            gate_dtype="float32",
            chunk_size=chunk,
            scale=dk**-0.5,
            block_S=chunk,
            block_DK=block_DK,
            block_DV=block_DV,
            threads=threads,
            num_stages=num_stages,
        )

    def inputs(device, generator):
        return [
            _gate(shape, chunk, device, generator) if i == 2 else _random(shape, dtype, device, generator) for i, shape in enumerate(shapes)
        ]

    return KernelCase(build, inputs, chunk_reference(w), [5])
