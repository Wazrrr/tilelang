"""TileTune numerical evaluation. Imports only the Python standard library."""

from .memory import classify_bound
from .ranking import rank_records, select_top_k, select_with_exploration
from .contracts import AnalysisReport, Diagnostic, KernelFacts
from .backends import BackendModel, backend_model, register_backend
from .evaluator import evaluate

__version__ = "0.1.0"


__all__ = [
    "classify_bound",
    "rank_records",
    "select_top_k",
    "select_with_exploration",
    "AnalysisReport",
    "Diagnostic",
    "KernelFacts",
    "BackendModel",
    "backend_model",
    "register_backend",
    "evaluate",
]
