"""Shards retain global identities and resume without repeating completed work."""

import json

import pytest


def test_census_resumes_interruption_preserves_indices_and_checks_provenance(tmp_path, monkeypatch):
    from experiments.utils import cli as _common
    from experiments.common import census, comparison as compare
    from experiments.utils.io import write_json
    from experiments.common.spec import Device, Workload, configuration_space
    from tilelang.cache.kernel_cache import KernelCache

    source = {"kernel.py": "original"}
    monkeypatch.setattr(_common, "source_hashes", lambda *args: dict(source))
    monkeypatch.setattr(KernelCache, "_get_tilelang_lib_stamp", lambda: "native-build")
    monkeypatch.setattr(compare, "wait_for_idle", lambda *args, **kwargs: None)
    calls = []

    def run_case(request, output):
        output.mkdir(parents=True)
        write_json(output / "request.json", request)
        (output / "worker.log").write_text("original worker log")
        indices = request["settings"]["config_indices"]
        calls.append(indices)
        if len(calls) == 2:
            raise RuntimeError("interrupted worker")
        space = configuration_space(Workload(**request["workload"]), Device(**request["device"]))
        records = [
            dict(
                index=local,
                original_index=index,
                config=space["configs"][index],
                config_id=space["config_ids"][index],
                status="benchmarked",
                latency_ms=0.01 + index / 10000,
                program_sha256="second" if index == 6 else "first",
            )
            for local, index in enumerate(indices)
        ]
        result = dict(status="completed", tuning_seconds=10)
        write_json(output / "outcomes.json", records)
        write_json(output / "result.json", result)
        return result

    monkeypatch.setattr(census, "run_case", run_case)
    root = tmp_path / "census"
    args = ["--workloads", "gemm_fp8_square", "--config-indices", "0", "6", "223", "--shard-size", "2", "--output", str(root)]
    with pytest.raises(RuntimeError, match="interrupted worker"):
        census.main(args)
    assert census.main(args + ["--resume"]) == 0
    assert calls == [[0, 6], [223], [223]]
    directory = root / "ampere" / "gemm_fp8_square"
    preserved = list(directory.glob("shard-00001.interrupted-*"))
    assert len(preserved) == 1
    assert (preserved[0] / "worker.log").read_text() == "original worker log"
    combined = json.loads((directory / "outcomes.json").read_text())
    assert [r["index"] for r in combined] == [0, 6, 223]
    assert [r["original_index"] for r in combined] == [0, 6, 223]
    assert [r["shard_index"] for r in combined] == [0, 1, 0]
    assert json.loads((directory / "shard-00001" / "outcomes.json").read_text())[0]["index"] == 0
    summary = json.loads((root / "summary.json").read_text())[0]
    assert summary["completed_count"] == summary["correct_count"] == 3
    assert summary["distinct_correct_program_count"] == 2
    assert summary["tuning_seconds"] == 20
    assert summary["current_best_ms"] is None
    assert summary["best_ms"] == 0.01
    stamps = {p: p.stat().st_mtime_ns for p in directory.glob("shard-*/*.json")}
    assert census.main(args + ["--resume"]) == 0
    assert len(calls) == 3
    assert stamps == {p: p.stat().st_mtime_ns for p in stamps}
    source["kernel.py"] = "changed"
    with pytest.raises(ValueError, match="source/compiler changed"):
        census.main(args + ["--resume"])
    with pytest.raises(ValueError, match="plan changed"):
        census.main(args + ["--resume", "--seed", "124"])
