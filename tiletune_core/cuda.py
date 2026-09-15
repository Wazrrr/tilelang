"""Compatibility evaluator for version 21 CUDA facts and equations.

The compiler adapter exports the existing resolved pipeline representation.
Archived reports remain readable; this extraction does not recalibrate rates.
"""

import time
from types import SimpleNamespace

from .contracts import AnalysisReport, Diagnostic
from .ranking import apply_ranking_metric, combine_tile_cost


def evaluate_cuda_facts(facts):
    if facts.backend != "cuda.v21":
        raise ValueError("expected CUDA version 21 facts")
    started = time.perf_counter()
    p = facts.payload
    cost = combine_tile_cost(p["memory"], p["waves"])
    ranking = apply_ranking_metric(
        cost,
        p["waves"],
        p["pipeline"],
        SimpleNamespace(**p["config"]),
        SimpleNamespace(matched=p["specialization_matched"]),
        p["register_demand"],
    )
    cost.update(score=ranking["score"], score_formula=ranking["formula"], ranking_metric=ranking["metric"])
    if ranking["score"] is None:
        cost["precision"] = "unknown"
        cost["unknown"] = sorted(set(cost["unknown"] + ranking["unknown"]))
    diagnostics = [Diagnostic("legacy_cuda_unknown", "uncertainty", reason) for reason in ranking["unknown"]]
    return AnalysisReport(
        facts.backend,
        ranking["score"],
        ranking.get("units", "cycles"),
        p["waves"],
        diagnostics,
        time.perf_counter() - started,
        dict(tile_cost=cost, ranking=ranking),
    )
