"""Inputs and reference for the SM100 BSHD MHA-forward example."""

from experiments.utils.kernel import KernelCase, _random

from threading import Lock
from .reference import reference_attention

_ATTENTION_LOCK = Lock()


def _attention_program(**kwargs):
    from examples.flash_attention_sm100.mha_fwd_bshd import flashattn

    with _ATTENTION_LOCK:
        return flashattn.get_tir(**kwargs)


def make_case(w):
    # Reuse the SM100 MHA-forward algorithm and its stable online softmax. The
    # eager builder is mutable, so serialize elaboration only.
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
            variant="ts" if threads == 256 else "ss",
            num_stages=num_stages,
            dtype=dtype,
        )

    def inputs(device, generator):
        return [_random((batch, sequence, heads, dim), w.dtype, device, generator) for _ in range(3)]

    return KernelCase(build, inputs, lambda q, k, v: reference_attention(q, k, v, causal).to(q.dtype), [3], {"tl.enable_fast_math": True})
