"""Inputs and reference for the BSHD FlashAttention example."""

from experiments.utils.kernel import KernelCase, _random

from threading import Lock
from .reference import reference_attention

_ATTENTION_LOCK = Lock()


def _attention_program(**kwargs):
    from examples.flash_attention.example_mha_fwd_bshd import flashattn

    with _ATTENTION_LOCK:
        return flashattn.jit_impl.get_tir(**kwargs)


def make_case(w):
    # Reuse the existing FlashAttention algorithm, including its stable online
    # softmax. The eager builder is mutable, so serialize elaboration only.
    p = w.parameters
    batch, heads, sequence, dim = (p[key] for key in ("batch", "heads", "sequence", "dim"))
    causal = p.get("causal", False)
    dtype = w.dtype

    def build(block_M, block_N, num_stages, threads):
        return _attention_program(
            batch=batch,
            heads=heads,
            seq_len=sequence,
            dim=dim,
            is_causal=causal,
            block_M=block_M,
            block_N=block_N,
            num_stages=num_stages,
            threads=threads,
            dtype=dtype,
        )

    def inputs(device, generator):
        return [_random((batch, sequence, heads, dim), w.dtype, device, generator) for _ in range(3)]

    return KernelCase(build, inputs, lambda q, k, v: reference_attention(q, k, v, causal).to(q.dtype), [3], {"tl.enable_fast_math": True})
