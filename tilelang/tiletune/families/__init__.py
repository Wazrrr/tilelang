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
    # A generic graph can contain a recurrent state update, several GEMMs, or
    # only elementwise/reduction operations. Select a loop only when unambiguous.
    loops = col.pipeline_loops or col.serial_loops
    loop = loops[0] if len(loops) == 1 else None
    return KernelSpecialization(
        loop=loop,
        matched=requested in ("auto", "generic"),
        evidence=[
            ("generic operation analysis with a unique loop" if loop is not None else "generic operation analysis")
            if requested in ("auto", "generic")
            else f"requested {requested} specialization did not match the actual operation graph"
        ],
    )
