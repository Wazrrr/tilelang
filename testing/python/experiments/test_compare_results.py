"""Offline comparisons preserve selection budgets and oracle measurement identity."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.compare_results import compare, render


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return path


@pytest.fixture
def inputs(tmp_path):
    oracle = [dict(index=i, config={"tile": i + 1}, status="benchmarked", latency_ms=ms) for i, ms in enumerate((1.0, 2.0, 4.0))] + [
        dict(index=3, config={"tile": 4}, status="compilation_failed", latency_ms=None)
    ]
    # Report order and local indices deliberately differ from oracle indices.
    records = [dict(index=i + 10, config=oracle[j]["config"]) for i, j in enumerate((2, 3, 1, 0))]
    report = dict(
        configs=records,
        ranking=[
            dict(index=11, tier="eligible", score=0),
            dict(index=10, tier="eligible", score=1),
            dict(index=12, tier="eligible", score=2),
            dict(index=13, tier="unknown", score=None),
        ],
        # An exploration shortlist can differ from the eligible ranking prefix.
        selection=dict(requested_k=2, selected_indices=[10, 13]),
    )
    return write(tmp_path / "oracle.json", oracle), write(tmp_path / "method.json", report), oracle, report


def test_failed_configs_consume_k_and_actual_shortlist_is_separate(inputs):
    oracle, method, _, _ = inputs
    result = compare(oracle, {"tiletune": method}, [1, 2, 3, 5])
    rows = result["methods"][0]["curves"]
    assert [r["oracle_at_k"] for r in rows] == [None, 0.25, 0.5, 0.5]
    assert rows[0]["status"] == "no_success"
    assert rows[1]["failed_count"] == 1
    assert rows[1]["best_index"] == 10 and rows[1]["best_oracle_index"] == 2
    assert rows[-1]["shortfall"] == 2
    saved = result["methods"][0]["saved_selection"]
    assert saved["selected_indices"] == [10, 13] and saved["oracle_at_k"] == 1
    assert saved["best_config"] == {"tile": 1}
    assert "100.00%" in render(result) and "N/A" in render(result)


def test_selection_only_reports_and_negative_xgboost_scores(inputs):
    oracle, method, _, report = inputs
    report["ranking"][0]["score"] = -5.0
    write(method, report)
    assert compare(oracle, {"xgboost": method}, [1])["methods"][0]["curves"][0]["oracle_at_k"] is None
    report.pop("ranking")
    write(method, report)
    with pytest.raises(ValueError, match="no saved ranking"):
        compare(oracle, {"carver": method}, [2])
    rows = compare(oracle, {"carver": method}, [1, 2, 5], "selected")["methods"][0]["curves"]
    assert [r["oracle_at_k"] for r in rows] == [0.25, 1, 1]
    assert rows[-1]["shortfall"] == 3


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_invalid_successful_oracle_latencies_are_not_admitted(inputs, value):
    oracle, method, records, _ = inputs
    records[0]["latency_ms"] = value
    write(oracle, records)
    with pytest.raises(ValueError, match="invalid oracle latency"):
        compare(oracle, {"tiletune": method}, [1])


@pytest.mark.parametrize("status", ["selected", "compiled", "not_attempted", "not_selected", "pending"])
def test_unfinished_oracles_are_rejected(inputs, status):
    oracle, method, records, _ = inputs
    records[3]["status"] = status
    write(oracle, records)
    with pytest.raises(ValueError, match="incomplete oracle outcome"):
        compare(oracle, {"tiletune": method}, [1])


def test_config_mismatch_and_duplicate_orders_are_rejected(inputs):
    oracle, method, _, report = inputs
    changed = deepcopy(report)
    changed["configs"][0]["config"] = {"tile": 999}
    write(method, changed)
    with pytest.raises(ValueError, match="missing from the oracle pool"):
        compare(oracle, {"tiletune": method}, [1])
    report["ranking"][1]["index"] = 11
    write(method, report)
    with pytest.raises(ValueError, match="duplicate candidate"):
        compare(oracle, {"tiletune": method}, [1])


def test_heuristic_resolves_verified_sweep_instead_of_remeasured_winner(inputs):
    oracle, method, records, _ = inputs
    write(oracle, dict(records=records, candidate_count=4))
    heuristic = write(
        oracle.parent / "heuristic.json",
        dict(winner=dict(latency_ms=99), reference=dict(path=oracle.name, sha256=hashlib.sha256(oracle.read_bytes()).hexdigest())),
    )
    result = compare(heuristic, {"tiletune": method}, [2])
    assert result["oracle"]["best_latency_ms"] == 1
    assert len(result["oracle"]["sources"]) == 2
    write(oracle, dict(records=records, candidate_count=5))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        compare(heuristic, {"tiletune": method}, [2])
    with pytest.raises(ValueError, match="candidate_count"):
        compare(oracle, {"tiletune": method}, [2])


def test_missing_candidate_is_detected_by_manifest_and_workload_mismatch(inputs):
    oracle, method, records, report = inputs
    manifest = dict(
        workload=dict(op="anything", dtype="float16", parameters=dict(rows=8)),
        device=dict(target=dict(kind="arbitrary-backend", arch="arbitrary-device")),
        configs=[r["config"] for r in records],
    )
    write(oracle.parent / "experiment.json", manifest)
    write(oracle, records[:-1])
    with pytest.raises(ValueError, match="records do not match experiment"):
        compare(oracle, {"tiletune": method}, [2])
    write(oracle, records)
    method = write(oracle.parent / "method" / "tiletune.json", report)
    changed = deepcopy(manifest)
    changed["workload"]["parameters"]["rows"] = 9
    write(method.parent / "experiment.json", changed)
    with pytest.raises(ValueError, match="workload differs"):
        compare(oracle, {"tiletune": method}, [2])
    write(method.parent / "experiment.json", manifest)
    assert compare(oracle, {"tiletune": method.parent}, [2])["methods"][0]["status"] == "evaluated"


def test_unavailable_carver_is_explicit_and_xgboost_is_optional(inputs):
    oracle, method, _, _ = inputs
    unavailable = write(method.parent / "carver.json", dict(status="unsupported", reason="no saved model for this operation"))
    result = compare(oracle, {"tiletune": method, "carver": unavailable}, [1, 2])
    assert len(result["methods"]) == 2
    assert result["methods"][1]["status"] == "unsupported"
    assert result["methods"][1]["curves"] == []


def test_embedded_oracle_identity_rejects_same_configs_for_another_shape(inputs):
    oracle, method, records, report = inputs
    identity = dict(workload=dict(op="gemm", dtype="float16", parameters=dict(m=4096, batch=1)))
    write(oracle, dict(records=records, candidate_count=4, identity=identity))
    method = write(oracle.parent / "method" / "tiletune.json", report)
    manifest = dict(workload=dict(op="gemm", dtype="float16", parameters=dict(m=8192)), configs=report["configs"])
    write(method.parent / "experiment.json", manifest)
    with pytest.raises(ValueError, match="workload differs"):
        compare(oracle, {"tiletune": method}, [2])
    manifest["workload"]["parameters"]["m"] = 4096
    write(method.parent / "experiment.json", manifest)
    assert compare(oracle, {"tiletune": method}, [2])["methods"][0]["status"] == "evaluated"


def test_null_selection_budget_has_a_clear_error(inputs):
    oracle, method, _, report = inputs
    report["selection"]["requested_k"] = None
    write(method, report)
    with pytest.raises(ValueError, match="invalid saved selection budget"):
        compare(oracle, {"tiletune": method}, [2])


def test_shell_entrypoint_runs_outside_repo_without_backend_packages(inputs, tmp_path):
    oracle, method, _, _ = inputs
    root = Path(__file__).resolve().parents[3]
    output = tmp_path / "comparison.json"
    run = subprocess.run(
        [
            "bash",
            str(root / "experiments/compare_results.sh"),
            "--oracle",
            str(oracle),
            "--tiletune",
            str(method),
            "--top-k",
            "1",
            "2",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Oracle@K" in run.stdout
    assert json.loads(output.read_text())["methods"][0]["curves"][1]["oracle_at_k"] == 0.25
    code = (
        f"import runpy, sys; sys.path.insert(0, {str(root)!r}); "
        f"runpy.run_path({str(root / 'experiments/compare_results.py')!r}, run_name='offline_import'); "
        "assert not {'torch', 'tilelang', 'tvm', 'numpy', 'xgboost'} & sys.modules.keys()"
    )
    subprocess.run([sys.executable, "-S", "-c", code], cwd=tmp_path, check=True)
