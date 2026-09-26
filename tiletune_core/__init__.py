"""TileTune numerical evaluation. Imports only the Python standard library."""

from .ranking import alpha_budget, rank_records, select_top_k, select_with_exploration
from .memory import classify_bound, score_memory, score_rank_product
from .work_max import score_work_max, score_work_rank_product
from .contracts import AnalysisReport, Diagnostic, KernelFacts
from .backends import BackendModel, backend_model, register_backend
from .evaluator import evaluate

__version__ = "0.1.0"


__all__ = [
    "rank_records",
    "alpha_budget",
    "select_top_k",
    "select_with_exploration",
    "classify_bound",
    "score_memory",
    "score_rank_product",
    "score_work_max",
    "score_work_rank_product",
    "AnalysisReport",
    "Diagnostic",
    "KernelFacts",
    "BackendModel",
    "backend_model",
    "register_backend",
    "evaluate",
]
