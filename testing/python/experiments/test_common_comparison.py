"""Disjoint workloads and honest oracle/coverage diagnostics for comparisons."""

from dataclasses import replace

import pytest

from experiments.common.baselines import carver_support_reason, exhaustive_selection
from experiments.common.comparison import main, split_workloads, training_sample
from experiments.utils.diagnostics import assess, spearman
from experiments.common.spec import Device, TARGETS, configurations, default_workloads


def test_split_whole_workloads_and_reject_alias_leakage():
    workloads = []
    for workload in default_workloads():
        if workload.op not in {w.op for w in workloads}:
            workloads.append(workload)
    splits = split_workloads(workloads, dict(train=[0.25, 0.5], validation=[0.75], test=[1, 2]))
    assert len(splits["test"]) == 2 * len(workloads)
    with pytest.raises(ValueError, match="overlap"):
        split_workloads([workloads[0], replace(workloads[0], name="alias")], dict(train=[0.5], test=[1]))
    with pytest.raises(ValueError, match="overlap"):
        split_workloads(workloads, dict(train=[1], test=[1]))


def test_training_collection_samples_before_measurement_and_preserves_original_indices():
    device = Device("ampere", TARGETS["ampere"])
    workload = default_workloads()[0]
    configs = configurations(workload, device)
    sample = training_sample(workload, device, fraction=0.1, seed=123)
    assert len(configs) == 576 and len(sample["config_indices"]) == 58
    assert sample["xgb_sampling"]["pool_configs"] == configs
    assert sample["config_indices"] == sample["xgb_sampling"]["selected_indices"]
    subset = [91, 40, 18, 72, 3, 5]
    sample = training_sample(workload, device, fraction=0.5, seed=123, config_indices=subset)
    assert len(sample["config_indices"]) == 3
    assert sample["xgb_sampling"]["pool_configs"] == [configs[i] for i in subset]
    assert sample["config_indices"] == [subset[i] for i in sample["xgb_sampling"]["selected_indices"]]


def test_comparison_freezes_training_fraction_separately_from_online_budget(capsys):
    import json

    for args, fraction in (([], 0.1), (["--xgb-sample-fraction", "0.25"], 0.25)):
        assert main(["--plan", "--workloads", "gemm_square", *args]) == 0
        plan = json.loads(capsys.readouterr().out)
        assert plan["xgb_sampling"]["fraction"] == fraction
        assert plan["xgb_sampling"]["seed"] == 123
        assert plan["budget_fraction"] == 0.1
        assert plan["xgb_training"] == dict(rounds=600, max_depth=10, learning_rate=0.05, subsample=0.8, early_stopping_rounds=20)
    with pytest.raises(SystemExit):
        main(["--plan", "--xgb-sample-fraction", "nan"])


def test_default_comparison_uses_the_same_disjoint_family_splits(capsys):
    import json
    from experiments.suite import study_plan

    assert main(["--plan", "--device", "hopper"]) == 0
    plan = json.loads(capsys.readouterr().out)
    expected = study_plan("full", [Device("hopper", TARGETS["hopper"])])
    assert plan["splits"] == expected["splits"]
    selected = ["gemm_square", "gemm_square_large"]
    assert main(["--plan", "--workloads", *selected]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert [w["name"] for w in plan["splits"]["test"]] == selected
    assert len(plan["splits"]["train"]) == 2
    assert len(plan["splits"]["validation"]) == 1


def test_diagnostics_keep_unscored_oracle_winner_and_failed_selections():
    report = dict(
        configs=[dict(index=i, config={}) for i in range(4)],
        ranking=[
            dict(index=0, score=1, tier="eligible"),
            dict(index=1, score=2, tier="eligible"),
            dict(index=2, score=3, tier="eligible"),
            dict(index=3, score=None, tier="unknown"),
        ],
        selection=dict(selected_indices=[0, 1]),
    )
    oracle = [
        dict(index=0, status="benchmark_error"),
        dict(index=1, status="benchmarked", latency_ms=2),
        dict(index=2, status="benchmarked", latency_ms=3),
        dict(index=3, status="benchmarked", latency_ms=1),
    ]
    result = assess(report, oracle)
    assert result["oracle_at_k"] == 0.5
    assert result["score_coverage"] == 0.75
    assert result["selected_oracle_successes"] == 1
    assert result["oracle_winner_rank"]["tier"] == "unknown"
    assert result["curves"]["1"]["oracle_at_k"] is None
    assert result["spearman_score_latency"] is None
    assert spearman([1, 2, 2, 4], [1, 2, 2, 4]) == pytest.approx(1)
    assert spearman([1, 1, 1], [1, 2, 3]) is None


def test_carver_unsupported_semantics_are_explicit_and_exhaustive_has_no_gate():
    device = Device("ampere", TARGETS["ampere"])
    workloads = {w.name: w for w in default_workloads()}
    assert carver_support_reason(workloads["gemm_square"], device) is None
    assert carver_support_reason(workloads["attention_noncausal"], device) is None
    assert carver_support_reason(workloads["grouped_gemm_aligned"], device) is None
    assert carver_support_reason(workloads["kda_intra_regular"], device) is None
    assert "FP8" in carver_support_reason(workloads["gemm_fp8_square"], device)
    for parameters in (dict(batch=2), dict(epilogue="bias_relu"), dict(transpose_b=False)):
        w = workloads["gemm_square"]
        assert carver_support_reason(replace(w, parameters=w.parameters | parameters), device)
    report = exhaustive_selection([dict(threads=128), dict(threads=99999)])
    assert report["selection"]["selected_indices"] == [0, 1]
    assert all(row["score"] is None for row in report["ranking"])
