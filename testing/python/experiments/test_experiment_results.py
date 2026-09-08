"""The measured winner must retain its model rank, ties, and uncertainty."""

import pytest

from experiments._new_carver import winner_summary
from tilelang.new_carver.ranking import rank_records


def report(scores):
    records = [
        {
            "index": i,
            "config": {"block": (i + 1) * 64},
            "tile_cost": {"score": score, "ranking_metric": "pipeline_time"},
            "latency_ms": 10 - i,
        }
        for i, score in enumerate(scores)
    ]
    return {"configs": records, "ranking": rank_records(records)}


def test_measured_winner_uses_model_rank_and_preserves_ties():
    data = report([30, 10, 20, 20])
    result = winner_summary(data, {"block": 256}, 0.5)
    assert result["index"] == 3
    assert result["latency_ms"] == 0.5
    assert result["predicted_rank"] == 3
    assert (result["tie_first_rank"], result["tie_last_rank"]) == (2, 3)
    assert result["scored_candidates"] == result["total_candidates"] == 4
    # Candidate measurements never alter the precomputed model ranking.
    for record in data["configs"]:
        record["latency_ms"] = record["index"] / 100
    assert winner_summary(data, {"block": 256}, 0.5) == result


@pytest.mark.parametrize("pressure_rejected", [False, True])
def test_unscored_or_rejected_winner_has_no_predicted_performance_rank(pressure_rejected):
    data = report([10, 20 if pressure_rejected else None])
    data["configs"][1]["pre_lowering"] = {"would_reject": pressure_rejected}
    data["ranking"] = rank_records(data["configs"])
    result = winner_summary(data, {"block": 128}, 0.5)
    assert result["predicted_rank"] is None
    assert result["tie_first_rank"] is result["tie_last_rank"] is None
    assert result["report_position"] == 2
    assert result["tier"] == ("pressure_rejected" if pressure_rejected else "unknown")
    assert result["scored_candidates"] == 1
    assert result["total_candidates"] == 2
