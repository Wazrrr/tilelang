"""Saved TileTune winner reporting for comparisons."""


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
        "uncertainty_first_rank": entry.get("uncertainty_first_rank") if scored else None,
        "uncertainty_last_rank": entry.get("uncertainty_last_rank") if scored else None,
        "report_position": entry["rank"],
        "tier": entry["tier"],
        "score_cycles": entry["score"],
        "scored_candidates": sum(row["tier"] == "eligible" and row["score"] is not None for row in report["ranking"]),
        "total_candidates": len(report["configs"]),
    }
