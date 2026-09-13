"""Disjoint workloads and honest oracle/coverage diagnostics for comparisons."""

from dataclasses import replace

import pytest

from experiments.portable.baselines import carver_support_reason, exhaustive_selection
from experiments.portable.compare import split_workloads
from experiments.portable.diagnostics import assess, spearman
from experiments.portable.spec import Device, TARGETS, default_workloads


def test_split_whole_workloads_and_reject_alias_leakage():
    workloads = default_workloads()
    splits = split_workloads(workloads, dict(train=[0.25, 0.5], validation=[0.75], test=[1, 2]))
    assert len(splits["test"]) == 2 * len(workloads)
    with pytest.raises(ValueError, match="overlap"):
        split_workloads([workloads[0], replace(workloads[0], name="alias")], dict(train=[0.5], test=[1]))
    with pytest.raises(ValueError, match="overlap"):
        split_workloads(workloads, dict(train=[1], test=[1]))


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
    assert carver_support_reason(workloads["gemm_nn"], device) is None
    for name in ("gemm_batched", "gemm_bias_relu", "flashattention", "softmax", "gemm_fp8"):
        assert carver_support_reason(workloads[name], device)
    report = exhaustive_selection([dict(threads=128), dict(threads=99999)])
    assert report["selection"]["selected_indices"] == [0, 1]
    assert all(row["score"] is None for row in report["ranking"])
