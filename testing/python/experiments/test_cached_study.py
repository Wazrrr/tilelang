"""Cache publication and contention evidence for the resumable study runner."""

from pathlib import Path

import pytest

from experiments.cached_study import find_bundle, successful_record
from experiments.utils.baseline_store import hash_files, publish_bundle
from experiments.utils.io import write_json


def identity():
    return dict(
        measurement=dict(kernel_contract_version=2, runtime=dict(device="NVIDIA H200", target=dict(kind="cuda", arch="sm_90a"))),
        splits=dict(test=[dict(name="test", op="gemm", parameters=dict(m=32, n=128, k=128), dtype="bfloat16")]),
        pools=dict(test=[dict(block_M=32)]),
    )


@pytest.mark.parametrize("status", ["contended", "running", "timeout", "worker_failed"])
def test_rejects_measurements_without_a_completed_contention_audit(tmp_path, status):
    write_json(tmp_path / "result.json", dict(status="completed"))
    assert not successful_record(tmp_path)
    write_json(tmp_path / "monitor.json", dict(status=status))
    assert not successful_record(tmp_path)
    write_json(tmp_path / "monitor.json", dict(status="uncontended"))
    assert successful_record(tmp_path)


def test_finds_prior_compatible_bundle_without_replacing_current(tmp_path):
    def save(name, ident):
        p = tmp_path / "runs" / name
        p.mkdir(parents=True)
        write_json(p / "outcomes.json", [1, 2])
        write_json(p / "complete.json", dict(identity=ident, artifacts=hash_files(p.glob("*.json"), p)))
        publish_bundle(tmp_path, p, ident)
        return p

    wanted = identity()
    good = save("old", wanted)
    changed = identity()
    changed["pools"]["test"].append(dict(block_M=64))
    save("new", changed)
    before = (tmp_path / "current.json").read_bytes()
    assert find_bundle(tmp_path, wanted) == good
    assert (tmp_path / "current.json").read_bytes() == before
    write_json(good / "outcomes.json", [3])
    with pytest.raises(ValueError, match="artifact changed"):
        find_bundle(tmp_path, wanted)


def test_plan_is_available_without_torch_or_compiler(tmp_path):
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[3]
    code = f"""
import sys
sys.path.insert(0, {str(root)!r})
from experiments.cached_study import main
assert main(['--output', {str(tmp_path)!r}, '--plan']) == 0
assert not any(n.split('.')[0] in ('torch', 'tilelang', 'xgboost') for n in sys.modules)
"""
    subprocess.run([sys.executable, "-I", "-S", "-c", code], check=True, capture_output=True)
    assert not list(tmp_path.iterdir())


def test_report_counts_fixed_baselines_once_and_preserves_failed_shortlists(tmp_path):
    from experiments.cached_report import report

    workload = dict(name="test", op="gemm", parameters=dict(m=32, n=128, k=128), dtype="bfloat16")
    write_json(tmp_path / "plan.json", dict(splits=dict(test=[workload]), budget=dict(seeds=[123, 456]), devices=[dict(name="hopper")]))
    write_json(tmp_path / "baselines.json", {})
    write_json(tmp_path / "profiles.json", {})
    for seed in (123, 456):
        case = tmp_path / "comparison" / str(seed) / "test"
        case.mkdir(parents=True)
        write_json(
            case / "comparison.json",
            dict(
                workload=workload,
                seed=seed,
                family="gemm",
                baseline_bundle=str(tmp_path / "bundle"),
                methods=dict(
                    brute_force=dict(status="completed", tuning_seconds=10), tiletune=dict(status="model_unavailable", tuning_seconds=1)
                ),
            ),
        )
        write_json(
            case / "oracle-curves.json",
            dict(
                oracle=dict(best_latency_ms=1, best_config=dict(block_M=32), statuses=dict(benchmarked=1)),
                methods=[dict(method="tiletune", curves=[], saved_selection=None)],
            ),
        )
    result = report(tmp_path)
    assert result["status"] == "completed"
    assert result["aggregate"]["brute_force"]["records"] == 1
    assert result["aggregate"]["brute_force"]["total_online_seconds"] == 10
    assert result["aggregate"]["tiletune"]["records"] == 2
    assert result["aggregate"]["tiletune"]["valid_records"] == 0
    assert result["aggregate"]["tiletune"]["geometric_mean_oracle_at_20"] is None
    assert (tmp_path / "comparison.csv").exists()
