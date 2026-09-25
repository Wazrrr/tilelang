"""Shared study contracts must not depend on a CUDA generation or branch history."""

from dataclasses import replace

import pytest

from experiments.common.run import make_request, validate_result
from experiments.common.smoke import instruction_evidence
from experiments.common.spec import Device, TARGETS, Workload
from experiments.families import FAMILIES, family_module
from experiments.suite import study_plan
from experiments.xgboost.data import canonical_workload, digest


@pytest.mark.parametrize("suite", ["development", "full", "final"])
def test_all_named_study_splits_are_mathematically_disjoint(suite):
    plan = study_plan(suite, [Device("ampere", TARGETS["ampere"])])
    keys = [digest(canonical_workload(w)) for values in plan["splits"].values() for w in values]
    assert len(keys) == len(set(keys))
    assert {w["op"] for w in plan["splits"]["test"]} == set(FAMILIES)


def test_planning_rejects_a_renamed_validation_test_overlap(monkeypatch):
    family = family_module("gemm", "cases")
    train_a, train_b, _ = family.training_cases()
    duplicate = replace(family.cases()[0], name="different_name")
    monkeypatch.setattr(family, "training_cases", lambda: [train_a, train_b, duplicate])
    with pytest.raises(ValueError, match="distinct mathematical shapes"):
        study_plan("development", [Device("ampere", TARGETS["ampere"])], families=["gemm"])


@pytest.mark.parametrize("method", ["carver", "top_k", "random"])
def test_ranked_workers_cannot_exceed_or_replace_the_frozen_shortlist(method):
    w = Workload("small", "gemm", dict(m=128, n=128, k=128, transpose_b=True))
    d = Device("hopper", TARGETS["hopper"])
    request = make_request(w, d, dict(method=method, top_k=2))
    result = dict(
        version=request["version"],
        request_id=request["request_id"],
        workload=w.name,
        device=d.name,
        status="completed",
        correctness="passed",
        winner=dict(index=0, latency_ms=1),
        selection=dict(requested_k=2, selected_count=2, selected_indices=[0, 1]),
        device_observation=dict(name="test", target=d.target),
    )
    assert validate_result(result, request) == result
    for indices in ([0, 1, 2], [1, 2]):
        result["selection"].update(selected_indices=indices, selected_count=len(indices))
        with pytest.raises(ValueError, match="selection budget"):
            validate_result(result, request)


def test_fp8_smoke_requires_matrix_and_fp8_instruction_evidence():
    for source, expected in (
        ("mma.sync.aligned.f32.f16.f16.f32;", "missing"),
        ("// mma.sync e4m3\nadd.f32;", "missing"),
        ("mma.sync.aligned.f32.e4m3.e4m3.f32;", "verified"),
        ("wgmma.mma_async.sync.aligned.f32.e5m2.e5m2;", "verified"),
    ):
        assert instruction_evidence(source, TARGETS["hopper"], "gemm_fp8", {})["status"] == expected


def test_empty_carver_selection_preserves_rejections_without_compiling(tmp_path):
    import json
    from types import SimpleNamespace
    from experiments.common.execution import run_selected

    def forbidden(**kwargs):
        raise AssertionError("an empty shortlist must not compile a replacement")

    configs = [{"block_M": 32}]
    report = dict(
        metric="carver_traffic_waves",
        score_units="byte-waves",
        selection=dict(requested_k=2, selected_count=0, selected_indices=[], wall_time_ms=0),
        ranking=[dict(index=0, rank=1, score=None, tier="unknown")],
        configs=[dict(index=0, config=configs[0], selected=False, status="model_rejected")],
    )
    result = run_selected(
        SimpleNamespace(build=forbidden), configs, [7], TARGETS["hopper"], [], [], {}, tmp_path, report, report_name="carver"
    )
    assert result["status"] == "model_unavailable" and "winner" not in result
    assert result["selection"]["selected_indices"] == []
    saved = json.loads((tmp_path / "carver.json").read_text())
    assert saved["configs"][0]["original_index"] == 7 and saved["configs"][0]["status"] == "model_rejected"
