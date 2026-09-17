"""Lazy experiment-family registry; planning never imports a compiler."""

from importlib import import_module

FAMILIES = {
    "gemm": "gemm",
    "attention": "flash_attention",
    "kda_chunk_o": "kda",
    "gemm_fp8": "gemm_fp8",
    "grouped_gemm": "grouped_gemm",
}

# The active matrix; archived studies retain their frozen manifests.
DEFAULT_OPS = tuple(FAMILIES)


def family_module(op, component):
    return import_module(f"experiments.{FAMILIES[op]}.{component}")
