"""Heuristics retain complete oracle outcomes and their actual timing protocol."""

import hashlib
import json
from pathlib import Path

import pytest

from experiments.common import brute_force
from experiments.common.spec import Device, TARGETS, Workload
from experiments.utils.io import write_json
from experiments.utils.results import load_oracle


@pytest.fixture
def saved_oracle(tmp_path, monkeypatch):
    monkeypatch.setattr(brute_force, "ROOT", tmp_path)
    workload = Workload("example", "gemm", dict(m=4, n=128, k=128, transpose_b=True))
    device = Device("ampere", TARGETS["ampere"])
    configs = [dict(BLOCK_M=i, BLOCK_N=128, threads=128) for i in (1, 2)]
    rows = {i: dict(index=i, config=c, config_id=str(i), status="benchmarked", latency_ms=2 - i) for i, c in enumerate(configs)}
    attempt = tmp_path / "baseline"
    attempt.mkdir()
    oracle = attempt / "outcomes.json"
    write_json(oracle, list(rows.values()))
    write_json(
        attempt / "experiment.json",
        dict(
            workload=workload.to_dict(),
            configs=configs,
            settings=dict(seed=456),
            device_observation=dict(
                name="NVIDIA A100",
                torch_version="torch",
                runtime_version="cuda",
                device_compiler_version="nvcc",
                host_compiler_version="g++",
            ),
            source_sha256={"kernel.py": "hash"},
            native_build="build",
            measurement_identity={"baseline": "identity"},
        ),
    )
    write_json(attempt / "result.json", dict(measurement=dict(backend="event", warmup=3, rep=17)))
    write_json(attempt / "monitor.json", dict(status="uncontended", gpus=[dict(uuid="GPU-test")]))
    validation = dict(validation=dict(brute_force=dict(median_ms=1.1, samples_ms=[1.1] * 7, relative_spread=0)))
    return tmp_path, workload, device, dict(configs=configs), rows, validation, oracle


def test_export_reuses_full_oracle_without_changing_baseline(saved_oracle):
    root, workload, device, space, rows, validation, oracle = saved_oracle
    before = {p.name: p.read_bytes() for p in oracle.parent.iterdir()}
    path = Path(brute_force.export_case(root, workload, device, space, rows, validation, root / "validation", oracle_path=oracle))
    heuristic = json.loads(path.read_text())
    assert heuristic["winner"]["index"] == 1
    assert heuristic["winner"]["sweep_latency_ms"] == 1
    assert heuristic["winner"]["latency_ms"] == 1.1
    assert heuristic["identity"]["measurement"] == dict(backend="event", warmup=3, rep=17, input_seed=456)
    assert heuristic["identity"]["measurement_identity"] == {"baseline": "identity"}
    assert heuristic["reference"]["sha256"] == hashlib.sha256(oracle.read_bytes()).hexdigest()
    assert load_oracle(path)["winner"]["latency_ms"] == 1
    assert {p.name: p.read_bytes() for p in oracle.parent.iterdir()} == before


@pytest.mark.parametrize("change", ["missing", "wrong_config", "unfinished"])
def test_export_rejects_incomplete_or_incompatible_pool(saved_oracle, change):
    root, workload, device, space, rows, validation, oracle = saved_oracle
    if change == "missing":
        rows.pop(1)
    elif change == "wrong_config":
        rows[1]["config"] = dict(BLOCK_M=4, BLOCK_N=128, threads=128)
    else:
        rows[1]["status"] = "compiled"
    with pytest.raises(ValueError, match="heuristic export requires"):
        brute_force.export_case(root, workload, device, space, rows, validation, root / "validation", oracle_path=oracle)
    assert not (root / "experiments").exists()


def test_standalone_oracle_accepts_ampere_fp8_storage_contract():
    from experiments.gemm_fp8.cases import cases
    from experiments.gemm_fp8.spaces import support_reason

    device = Device("ampere", TARGETS["ampere"])
    assert all(support_reason(workload, device) is None for workload in cases(holdout=True))
