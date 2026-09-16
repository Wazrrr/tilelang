"""Lazy experiment-family registry; planning never imports a compiler."""

from importlib import import_module

FAMILIES = {
    "gemm": "gemm",
    "attention": "flash_attention",
    "kda_chunk_o": "kda",
    "softmax": "softmax",
    "grouped_gemm": "grouped_gemm",
}

# Preserve the frozen four-family study; additional families are opt-in.
DEFAULT_OPS = ("gemm", "attention", "kda_chunk_o", "softmax")


def family_module(op, component):
    return import_module(f"experiments.{FAMILIES[op]}.{component}")
