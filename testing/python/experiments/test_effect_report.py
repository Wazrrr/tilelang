"""Cost and coverage semantics for the cross-method effect report."""

import pytest

from experiments.effect_report import _cost, _method_summary


def test_method_cost_excludes_optional_shortlist_validation():
    oracle = {"summary": {"tuning_seconds": 100.0}, "configs": [{} for _ in range(10)]}
    cost = _cost(
        oracle,
        selected_count=2,
        selection_seconds=3.0,
        preparation={"label_collection_seconds": 20.0, "cpu_training_seconds": 2.0},
    )
    assert cost["method_cost_seconds"] == 25.0
    assert cost["method_cost_vs_exhaustive_percent"] == 25.0
    assert cost["optional_shortlist_validation_gpu_estimated_seconds"] == 20.0
    assert cost["end_to_end_with_optional_validation_seconds"] == 45.0

    analytical = _cost(oracle, selected_count=2, selection_seconds=3.0, preparation={})
    assert analytical["method_cost_seconds"] == 3.0
    assert analytical["preparation_gpu_estimated_seconds"] == 0.0


def test_workload_coverage_is_distinct_from_quality_pass_rate():
    available_cost = {
        "method_cost_seconds": 2.0,
        "source_exhaustive_one_gpu_seconds": 10.0,
        "selection_cpu_measured_seconds": 2.0,
        "preparation_gpu_estimated_seconds": 0.0,
        "preparation_cpu_measured_seconds": 0.0,
        "optional_shortlist_validation_gpu_estimated_seconds": 1.0,
    }
    rows = [
        {
            "status": "available",
            "effect": {"passes_under_50_percent": True, "latency_gap_percent": 10.0, "oracle_at_k": 0.9},
            "cost": available_cost,
        },
        {
            "status": "available",
            "effect": {"passes_under_50_percent": False, "latency_gap_percent": 60.0, "oracle_at_k": 0.6},
            "cost": available_cost,
        },
        {"status": "unavailable"},
    ]
    summary = _method_summary(rows, total_workloads=3)
    assert summary["coverage"] == "2/3"
    assert summary["passing_workloads"] == 1
    assert summary["pass_rate_available_percent"] == 50.0
    assert summary["pass_rate_all_workloads_percent"] == pytest.approx(100 / 3)
