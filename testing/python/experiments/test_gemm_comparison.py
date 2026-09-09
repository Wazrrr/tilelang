"""Common-grid evaluation must preserve model choices and identify coverage."""

import json

from experiments._common import write_json
from experiments.gemm.tiletune.run import compare_results


def test_comparison_uses_original_ids_and_never_changes_selection(tmp_path):
    outputs = {method: tmp_path / method for method in ("brute_force", "carver", "tiletune")}
    for directory in outputs.values():
        directory.mkdir()
    write_json(
        outputs["brute_force"] / "outcomes.json",
        [dict(index=i, status="benchmarked", latency_ms=latency) for i, latency in enumerate([3, 1, 2])],
    )
    summaries = [
        dict(method=method, winner={"index": index}, tuning_seconds=seconds, validation_latency_ms=latency)
        for method, index, seconds, latency in [("brute_force", 1, 9, 1), ("carver", 0, 3, 3), ("tiletune", 1, 4, 1)]
    ]
    for method, ordering in [("carver", [0, 2, 1]), ("tiletune", [1, 2, 0])]:
        report = dict(
            selection={"selected_indices": ordering[:1]},
            ranking=[
                dict(index=i, rank=rank, score=rank, tier="eligible", tie_first_rank=rank, tie_last_rank=rank)
                for rank, i in enumerate(ordering, 1)
            ],
        )
        write_json(outputs[method] / f"{method}.json", report)
    before = (outputs["carver"] / "carver.json").read_bytes()
    compare_results(summaries, outputs)
    assert summaries[1]["top_k_oracle_retained_performance"] == 1 / 3
    assert summaries[2]["top_k_oracle_retained_performance"] == 1
    assert summaries[1]["brute_force_winner_model_rank"]["predicted_rank"] == 3
    assert not summaries[1]["brute_force_winner_model_rank"]["selected"]
    assert summaries[2]["brute_force_winner_model_rank"]["selected"]
    assert (outputs["carver"] / "carver.json").read_bytes() == before
    assert json.loads(before)["selection"]["selected_indices"] == [0]
