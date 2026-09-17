"""Lazy experiment-family registry; planning never imports a compiler."""

from importlib import import_module

FAMILIES = {
    "gemm": "gemm",
    "attention": "flash_attention",
    "kda_chunk_o": "kda",
    "gemm_fp8": "gemm_fp8",
    "grouped_gemm": "grouped_gemm",
}

# The complete study covers every registered kernel family. Keep this ordering
# stable because it also defines deterministic suite and report ordering.
DEFAULT_OPS = ("gemm", "attention", "kda_chunk_o", "gemm_fp8", "grouped_gemm")


def family_module(op, component):
    return import_module(f"experiments.{FAMILIES[op]}.{component}")
