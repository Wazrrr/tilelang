"""Lazy experiment-family registry; planning never imports a compiler."""

from importlib import import_module

FAMILIES = {
    "gemm": "gemm",
    "attention": "flash_attention",
    "kda_chunk_o": "kda",
    "gemm_fp8": "gemm_fp8",
    "grouped_gemm": "grouped_gemm",
}

# The core study replaces softmax with the repository's tensor-core FP8 GEMM.
# Grouped GEMM remains opt-in because its oracle is substantially larger.
DEFAULT_OPS = ("gemm", "attention", "kda_chunk_o", "gemm_fp8")


def family_module(op, component):
    return import_module(f"experiments.{FAMILIES[op]}.{component}")
