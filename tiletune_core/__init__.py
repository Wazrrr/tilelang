"""TileTune numerical evaluation. Imports only the Python standard library."""

from .ranking import rank_records, select_top_k, select_with_exploration
from .memory import score_memory
from .contracts import AnalysisReport, Diagnostic, KernelFacts
from .backends import BackendModel, backend_model, register_backend
from .evaluator import evaluate

__version__ = "0.1.0"


__all__ = [
    "rank_records",
    "select_top_k",
    "select_with_exploration",
    "score_memory",
    "AnalysisReport",
    "Diagnostic",
    "KernelFacts",
    "BackendModel",
    "backend_model",
    "register_backend",
    "evaluate",
]
