"""Recognize the actual operation graph and select its analysis policies.

Attention is checked first, then single-GEMM kernels, then generic operations.
An explicitly requested family must still match the graph; it is not a template.
"""

from .attention import AttentionSpecialization
from .base import KernelSpecialization
from .gemm import GemmSpecialization

__all__ = ["AttentionSpecialization", "GemmSpecialization", "KernelSpecialization", "select_specialization"]


def select_specialization(col, requested="auto"):
    for implementation in (AttentionSpecialization, GemmSpecialization):
        result = implementation.match(col)
        if result is not None and requested in ("auto", result.name):
            return result
    return KernelSpecialization(
        matched=requested in ("auto", "generic"),
        evidence=[
            "generic operation analysis"
            if requested in ("auto", "generic")
            else f"requested {requested} specialization did not match the actual operation graph"
        ],
    )
