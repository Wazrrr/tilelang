"""Repeated experiments preserve old data and keep comparisons in one run."""

import json

import pytest


from experiments.utils.io import write_json
from experiments.common import system
from experiments.utils import monitor


def test_all_variants_share_one_version_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    gpus = [dict(index=str(i), uuid=f"GPU-{i}", name="H200", **{"utilization.gpu": "0"}) for i in range(2)]
    monkeypatch.setattr(monitor, "snapshot", lambda: dict(gpus=gpus, processes=[]))

    def write_child_results(command, output, active, **kwargs):
        request = json.loads((output / "request.json").read_text())
        variant = request["variant"]
        assert len(active) == (2 if system.VARIANTS[variant][2] else 1)
        assert request["indices"] == [0, 1]
        write_json(
            output / "summary.json", dict(variant=variant, workload=request["workload"]["name"], tuning_seconds=1.0, winner_latency_ms=0.1)
        )

    monkeypatch.setattr(monitor, "run_monitored", write_child_results)
    root = tmp_path / "run"
    assert system.main(["--family", "gemm", "--output", str(root), "--config-indices", "0", "1"]) == 0
    comparison = json.loads((root / "comparison.json").read_text())
    assert len(comparison) == 10
    assert all((root / row["workload"] / row["variant"] / "summary.json").is_file() for row in comparison)
    with pytest.raises(FileExistsError):
        system.main(["--family", "gemm", "--output", str(root)])
