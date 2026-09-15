"""FlashAttention inputs, validation, and construction for both experiments."""

from threading import Lock

from examples.flash_attention.example_mha_tiletune import (
    PASS_CONFIGS as PASS_CONFIGS,
    make_attention,
    make_inputs as make_inputs,
)


_ELABORATION_LOCK = Lock()


def make_kernel(batch, heads, sequence, dim, causal):
    def attention(block_M, block_N, num_stages, threads):
        with _ELABORATION_LOCK:
            return make_attention(batch, heads, sequence, dim, causal)(block_M, block_N, num_stages, threads)

    return attention
