"""Compatibility imports; family implementations now live in ``families/``.

New code should import families from that package and IR helpers from ir_utils.
"""

from .families import AttentionSpecialization, GemmSpecialization, KernelSpecialization, select_specialization
from .ir_utils import call_names, dense_gemms, in_loop, main_loops

__all__ = [
    "AttentionSpecialization",
    "GemmSpecialization",
    "KernelSpecialization",
    "select_specialization",
    "call_names",
    "dense_gemms",
    "in_loop",
    "main_loops",
]
