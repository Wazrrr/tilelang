"""Lazy experiment-family registry; planning never imports a compiler."""

from importlib import import_module

FAMILIES = {
    "gemm": "gemm",
    "attention": "flash_attention",
    "kda_recurrent": "kda",
    "kda_chunk_o": "kda",
    "softmax": "softmax",
    "rmsnorm": "vector",
    "reduce_sum": "vector",
    "elementwise": "vector",
}


def family_module(op, component):
    return import_module(f"experiments.{FAMILIES[op]}.{component}")
