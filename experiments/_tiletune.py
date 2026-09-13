"""Shared TileTune arguments and model-rank reporting for comparisons."""

from pathlib import Path

from experiments._common import add_run_arguments, positive_int


def add_arguments(parser, family):
    add_run_arguments(parser, f"experiments/results/{family}/tiletune")
    parser.add_argument("--device-profile", type=Path, help="Reusable primitive profile; measure missing rates before tuning")
    parser.add_argument("--memory-regime", choices=["streaming", "cached"], default="streaming")
    parser.add_argument("--group-size", type=positive_int, default=1, help="Configurations per compilation unit; 1 disables grouping")


def winner_summary(report, winner_config, latency_ms):
    """Look up the measured winner in the model ranking, retaining unknowns and ties."""
    record = next(record for record in report["configs"] if record["config"] == winner_config)
    entry = next(entry for entry in report["ranking"] if entry["index"] == record["index"])
    scored = entry["tier"] == "eligible" and entry["score"] is not None
    return {
        "config": winner_config,
        "index": record["index"],
        "latency_ms": latency_ms,
        "predicted_rank": entry["rank"] if scored else None,
        "tie_first_rank": entry["tie_first_rank"] if scored else None,
        "tie_last_rank": entry["tie_last_rank"] if scored else None,
        "report_position": entry["rank"],
        "tier": entry["tier"],
        "score_cycles": entry["score"],
        "scored_candidates": sum(row["tier"] == "eligible" and row["score"] is not None for row in report["ranking"]),
        "total_candidates": len(report["configs"]),
    }


def rank_info(report, index):
    entry = next(row for row in report["ranking"] if row["index"] == index)
    scored = entry["tier"] == "eligible" and entry["score"] is not None
    return dict(
        predicted_rank=entry["rank"] if scored else None,
        tie_first_rank=entry["tie_first_rank"] if scored else None,
        tie_last_rank=entry["tie_last_rank"] if scored else None,
        tier=entry["tier"],
        selected=report["selection"] is None or index in report["selection"]["selected_indices"],
    )
