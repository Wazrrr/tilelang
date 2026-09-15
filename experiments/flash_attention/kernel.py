"""Family-owned portable implementations and independent references."""

from experiments._kernel import KernelCase, _random

from threading import Lock
from .kernels.baseline import make_kernel as make_kernel, make_inputs as make_inputs, PASS_CONFIGS as PASS_CONFIGS
from .reference import reference as reference, check_accuracy as check_accuracy
from .spaces import legacy_configurations

_ATTENTION_LOCK = Lock()


def _attention_program(**kwargs):
    from examples.flash_attention.example_mha_fwd_bshd import flashattn

    with _ATTENTION_LOCK:
        return flashattn.jit_impl.get_tir(**kwargs)


def attention_case(w):
    # Reuse the existing FlashAttention algorithm, including its stable online
    # softmax. The eager builder is mutable, so serialize elaboration only.
    from .reference import reference_attention

    p = w.parameters
    batch, heads, sequence, dim = (p[key] for key in ("batch", "heads", "sequence", "dim"))
    causal = p.get("causal", False)
    dtype = w.dtype

    def build(
        block_M, block_N, num_stages, threads, qk_policy="full_row", pv_policy="full_row", copy_width=None, implementation="baseline"
    ):
        if implementation not in ("baseline", "tiled"):
            raise ValueError("attention implementation must be baseline or tiled")
        if implementation == "tiled" or qk_policy != "full_row" or pv_policy != "full_row" or copy_width is not None:
            from .kernels.tiled import attention_program

            return attention_program(
                batch, heads, sequence, dim, causal, dtype, block_M, block_N, num_stages, threads, qk_policy, pv_policy, copy_width
            )
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


# Existing fixed-grid experiment API.

make_case = attention_case


def get_configs():
    return legacy_configurations()
