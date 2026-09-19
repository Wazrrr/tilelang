"""Larger budgets keep pool denominators, seed coverage and frozen provenance."""

from copy import deepcopy
import json

import pytest

from experiments.compare_results import compare
from experiments.topk_study import aggregate, budget_levels, render, sweep


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return path


def test_pool_percent_rounds_up_and_all_hit_cutoff_uses_every_record():
    assert budget_levels(192, [100], [20]) == {"K=100": 100, "20%": 39}
    rows = [
        dict(workload="a", seed=123, pool_size=720, first_oracle_hit_k=29),
        dict(workload="a", seed=456, pool_size=720, first_oracle_hit_k=31),
    ]
    result = aggregate(rows)
    assert result["all_hit_k"] == 31
    assert result["all_hit_whole_pool_percent"] == 5
    assert aggregate(rows[:1])["all_hit_whole_pool_percent"] == 4  # ceil(0.04 * 720) == 29
    rows[-1]["first_oracle_hit_k"] = None
    result = aggregate(rows)
    assert result["reachable_records"] == 1
    assert result["all_hit_k"] is None
    assert result["all_hit_whole_pool_percent"] is None
    assert result["unreachable"] == [dict(workload="a", seed=456)]


@pytest.fixture
def study(tmp_path):
    oracle = write(
        tmp_path / "oracle.json",
        [
            dict(index=0, config={"tile": 1}, status="compilation_failed", latency_ms=None),
            dict(index=1, config={"tile": 2}, status="benchmarked", latency_ms=2.0),
            dict(index=2, config={"tile": 3}, status="benchmarked", latency_ms=1.0),
        ],
    )
    method = dict(
        configs=[dict(index=i + 10, config={"tile": i + 1}) for i in range(3)],
        ranking=[dict(index=i + 10, tier="eligible", score=i) for i in range(3)],
        selection=dict(requested_k=1, selected_indices=[10]),
    )
    xgboost = write(tmp_path / "xgboost.json", method)
    excluded = deepcopy(method)
    excluded["ranking"][-1].update(tier="unknown", score=None)
    excluded["configs"][-1]["diagnostics"] = [dict(reason="unresolved pipeline")]
    tiletune = write(tmp_path / "tiletune.json", excluded)
    curves = compare(oracle, dict(tiletune=tiletune, xgboost=xgboost), [20])
    root = tmp_path / "study"
    write(root / "comparison.json", dict(status="completed", expected_comparisons=2))
    for seed in (123, 456):
        case = root / "comparison" / str(seed) / "shape"
        write(case / "oracle-curves.json", curves)
        write(
            case / "comparison.json",
            dict(family="gemm", seed=seed, workload=dict(name="shape", dtype="float16", parameters={})),
        )
    return root, tiletune


def test_sweep_keeps_all_tiletune_seeds_and_deduplicates_fixed_baseline(study):
    root, _ = study
    result = sweep(root, [1, 3, 100], [20], ["tiletune", "xgboost"])
    tt, xgb = result["aggregate"]["tiletune"], result["aggregate"]["xgboost"]
    assert tt["records"] == 2 and tt["reachable_records"] == 0
    assert tt["all_hit_k"] is None
    assert xgb["records"] == 1 and xgb["all_hit_k"] == 3
    assert xgb["all_hit_whole_pool_percent"] == 67
    assert result["rows"][0]["budgets"]["20%"]["k"] == 1
    assert result["rows"][0]["budgets"]["100%"]["shortfall"] == 1
    assert result["rows"][0]["budgets"]["100%"]["oracle_at_k"] == 0.5
    assert "unresolved pipeline" in render(result)
    assert "Unreachable" in render(result)


def test_sweep_rejects_changed_ranking_or_incomplete_study(study):
    root, tiletune = study
    changed = json.loads(tiletune.read_text())
    changed["ranking"][0]["score"] = -1
    write(tiletune, changed)
    with pytest.raises(ValueError, match="ranking changed"):
        sweep(root, [100], [20], ["tiletune"])
    write(root / "comparison.json", dict(status="incomplete", expected_comparisons=3))
    with pytest.raises(ValueError, match="completed study"):
        sweep(root, [100], [20], ["tiletune"])
