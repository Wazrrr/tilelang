"""Shared-oracle audits work across seed directories and preserve config identity."""

import hashlib
import json
import os

import pytest

from experiments.common.audit_model import audit


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return path


@pytest.fixture
def saved_case(tmp_path):
    oracle = write(
        tmp_path / "oracle/device/test/case/brute_force/outcomes.json",
        [
            dict(index=0, config={"tile": 64}, status="benchmarked", latency_ms=1),
            dict(index=1, config={"tile": 32}, status="benchmarked", latency_ms=2),
        ],
    )
    directory = tmp_path / "123/device/device/test/case"
    report = write(
        directory / "tiletune/tiletune.json",
        dict(
            analysis_version=1,
            selection=dict(requested_k=1, selected_indices=[10]),
            configs=[
                dict(
                    index=i,
                    config={"tile": tile},
                    ranking=dict(tier=tier),
                    modules=dict(pipeline_overlap=dict(phases=[]), waves={}),
                    pre_lowering={},
                    pressure=dict(register_demand=dict(status="predicted")),
                    tile_propagation=dict(operations=[]),
                )
                for i, tile, tier in [(10, 32, "eligible"), (11, 64, "unknown")]
            ],
        ),
    )
    comparison = write(
        directory / "comparison.json",
        dict(
            workload=dict(name="case", op="test"),
            device="device",
            oracle=dict(path=os.path.relpath(oracle, directory), sha256=hashlib.sha256(oracle.read_bytes()).hexdigest()),
        ),
    )
    return tmp_path, report, oracle, comparison


def test_nested_study_uses_referenced_oracle_and_matches_configs_not_indices(saved_case):
    root, _, _, _ = saved_case
    result = audit(root)
    case = result["cases"][0]
    assert result["version"] == 2
    assert case["winner"] == 11 and case["winner_config"] == {"tile": 64}
    assert case["best_selected"] == 10 and case["oracle_at_k"] == 0.5
    assert case["coverage_ceiling"] == 0.5 and case["ranking_within_eligible"] == 1


def test_changed_oracle_is_rejected(saved_case):
    root, _, oracle, _ = saved_case
    oracle.write_text(oracle.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        audit(root)


def test_archived_reports_require_explicit_oracle_instead_of_guessing_layout(saved_case):
    root, report, oracle, comparison = saved_case
    data = json.loads(comparison.read_text())
    data.pop("oracle")
    write(comparison, data)
    with pytest.raises(ValueError, match="no oracle reference"):
        audit(root)
    result = audit(root, report_path=report, oracle_path=oracle)
    assert result["cases"][0]["oracle_at_k"] == 0.5
