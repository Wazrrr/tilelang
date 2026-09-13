"""FlashAttention inputs, validation, and construction for both experiments."""

from threading import Lock

from examples.flash_attention.example_mha_tiletune import (
    PASS_CONFIGS as PASS_CONFIGS,
    check_accuracy as _check_accuracy,
    get_configs as get_configs,
    make_attention,
    make_inputs as make_inputs,
    reference_attention as _reference_attention,
)


_ELABORATION_LOCK = Lock()


def make_kernel(batch, heads, sequence, dim, causal):
    def attention(block_M, block_N, num_stages, threads):
        with _ELABORATION_LOCK:
            return make_attention(batch, heads, sequence, dim, causal)(block_M, block_N, num_stages, threads)

    return attention


def check_accuracy(actuals, refs):
    _check_accuracy(actuals[0], refs[0])


def reference(q, k, v, causal):
    return _reference_attention(q, k, v, causal)
