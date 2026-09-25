"""Baseline reuse, system ablations and contention rejection without a GPU."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.utils import baseline_store, monitor
from experiments.common import study
from experiments.utils.results import check_metadata
from experiments.utils.io import write_json
from experiments.common.spec import Device, TARGETS, configuration_space
from experiments.common.system import VARIANTS, system_plan
from experiments.suite import study_plan


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, data)


@pytest.mark.parametrize(
    "family,count", [("gemm", 576), ("flash_attention", 480), ("kda", 234), ("gemm_fp8", 2304), ("grouped_gemm", 576)]
)
def test_system_ablations_share_final_cases_and_full_ordered_pool(family, count):
    plan = system_plan(family)
    assert len(plan) == 25
    assert len({row["workload"]["name"] for row in plan}) == 5
    assert {row["variant"] for row in plan} == set(VARIANTS)
    assert all(
        row["indices"] == list(range(count)) and row["workload"]["dtype"] == ("float8_e4m3fn" if family == "gemm_fp8" else "bfloat16")
        for row in plan
    )
    assert system_plan(family, variants=["combined"], indices=[3, 1])[0]["indices"] == [3, 1]


def test_only_measurement_changes_invalidate_source_identity(tmp_path):
    for name in ("tilelang/tiletune/ranking.py", "tiletune_core/ranking.py", "src/lower.cc", "experiments/gemm_fp8/kernel.py"):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("original")
    before = baseline_store.measurement_sources(["gemm_fp8"], tmp_path)
    for name in ("tilelang/tiletune/ranking.py", "tiletune_core/ranking.py"):
        (tmp_path / name).write_text("new ranking")
    assert baseline_store.measurement_sources(["gemm_fp8"], tmp_path) == before
    (tmp_path / "experiments/gemm_fp8/kernel.py").write_text("new kernel")
    assert baseline_store.measurement_sources(["gemm_fp8"], tmp_path) != before
    assert "src/lower.cc" in before


def fixture_plan():
    device = Device("hopper", TARGETS["hopper"], performance_model={"test": 1})
    plan = study_plan("full", [device], families=["gemm_fp8"])
    plan["splits"]["test"] = plan["splits"]["test"][:1]
    plan["budget"]["seeds"] = [123]
    from experiments.common.spec import Workload

    w = Workload(**plan["splits"]["test"][0])
    configs = configuration_space(w, device)["configs"][:2]
    device = Device("hopper", TARGETS["hopper"], performance_model={"test": 1}, configs={w.name: configs})
    plan["devices"] = [device.to_dict()]
    return plan, device, configs


def test_runtime_probe_does_not_serialize_full_candidate_pools(monkeypatch):
    plan = study_plan("full", [Device("ampere", TARGETS["ampere"])])
    device = Device(**plan["devices"][0])
    assert len(json.dumps(device.to_dict())) > 131072

    def probe(argv, **kwargs):
        assert json.loads(argv[-1]) == dict(name=device.name, target=device.target)
        assert len(argv[-1]) < 1024
        return '{"device": "test"}'

    monkeypatch.setattr(baseline_store.subprocess, "check_output", probe)
    monkeypatch.setattr(monitor, "select_cuda_gpu", lambda device: {"uuid": "GPU-test"})
    assert baseline_store.runtime_identity(device) == {"device": "test"}


def test_unsupported_fp8_study_retains_every_case_and_seed(tmp_path, monkeypatch):
    import experiments.suite as suite

    plan = study_plan("full", [Device("ampere", TARGETS["ampere"])], families=["gemm_fp8"])
    monkeypatch.setattr(study, "runtime_identity", lambda device: {})

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported FP8 must not collect baselines, profile, or launch kernels")

    monkeypatch.setattr(study, "collect_bundle", forbidden)
    monkeypatch.setattr(suite, "_existing_or_run", forbidden)
    assert study.execute(plan, tmp_path, {}, baseline_root=tmp_path / "baselines") == 1
    references = json.loads((tmp_path / "baselines.json").read_text())
    assert references["ampere"]["gemm_fp8"]["status"] == "unsupported"
    for seed in plan["budget"]["seeds"]:
        rows = json.loads((tmp_path / str(seed) / "ampere/comparison.json").read_text())["results"]
        assert len(rows) == 5
        for row in rows:
            assert set(row["methods"]) == set(plan["methods"])
            assert all(method["status"] == "unsupported" and "FP8" in method["reason"] for method in row["methods"].values())
    acceptance = json.loads((tmp_path / "acceptance.json").read_text())
    assert not acceptance["accepted"]
    assert all(len(seed["cases"]) == 5 for seed in acceptance["targets"]["ampere"]["seeds"].values())


def test_baseline_identity_ignores_tiletune_seed_and_k_but_includes_baseline_seed(monkeypatch):
    plan, device, _ = fixture_plan()
    settings = dict(warmup=1, rep=2, timeout=30, case_timeout=300, workers=1)
    monkeypatch.setattr(baseline_store, "measurement_sources", lambda families: {"kernel": "hash"})
    _, first = baseline_store.identities(plan, device, settings, {"device": "test"})
    plan["top_k"] = 50
    plan["budget"]["seeds"] = [456]
    _, second = baseline_store.identities(plan, device, settings, {"device": "test"})
    assert first == second
    assert baseline_store.identities(plan, device, settings, {"device": "test"}, 456)[1] != first
    assert baseline_store.identities(plan, device, settings | {"rep": 3}, {"device": "test"})[1] != first


def ranking(configs):
    return dict(
        configs=[dict(index=i, config=c, status="benchmarked", latency_ms=i + 1) for i, c in enumerate(configs)],
        ranking=[dict(index=i, rank=i + 1, score=i + 1, tier="eligible") for i in range(len(configs))],
        selection=dict(requested_k=20, selected_indices=list(range(len(configs)))),
    )


@pytest.mark.parametrize("carver_status", ["unsupported", "model_unavailable"])
def test_two_tiletune_runs_collect_baselines_once_and_preserve_artifacts(tmp_path, monkeypatch, carver_status):
    import experiments.suite as suite

    plan, device, configs = fixture_plan()
    workload = plan["splits"]["test"][0]
    measurement = {"compiler": "fixed", "kernel": "fixed"}
    identity = dict(splits=plan["splits"], pools={workload["name"]: configs}, baseline_seed=123)
    settings = dict(warmup=1, rep=2, timeout=30, case_timeout=300, workers=1)
    calls = dict(collection=0, tiletune=0, remeasure=0)
    monkeypatch.setattr(study, "runtime_identity", lambda device: {})
    monkeypatch.setattr(study, "identities", lambda *args: (measurement, identity))

    def collect(command, *, check):
        calls["collection"] += 1
        assert check and command[command.index("--methods") + 1 : command.index("--methods") + 3] == ["carver", "xgboost"]
        base = Path(command[command.index("--output") + 1]) / device.name / "test" / workload["name"]
        report = ranking(configs)
        success = dict(status="completed", correctness="passed", tuning_seconds=10, winner=dict(config=configs[0], index=0))
        methods = dict(brute_force=success, xgboost=success, carver=dict(status=carver_status, reason="no eligible Carver candidates"))
        write(base / "methods.json", methods)
        for method, result in methods.items():
            write(base / method / "result.json", result)
            if method != "carver":
                write(base / method / (method + ".json"), report)
                write(base / method / "experiment.json", dict(workload=workload, configs=configs, measurement_identity=measurement))
        write(base / "brute_force/outcomes.json", report["configs"])
        if carver_status == "model_unavailable":
            rejected = deepcopy(report)
            rejected["selection"]["selected_indices"] = []
            for row in rejected["configs"]:
                row.update(status="model_rejected")
                del row["latency_ms"]
            for row in rejected["ranking"]:
                row.update(tier="unknown", score=None)
            write(base / "carver/carver.json", rejected)

    monkeypatch.setattr(
        study,
        "collect_bundle",
        lambda root, identity, measurement, device, settings: baseline_store.collect_bundle(
            root, identity, measurement, device, settings, run=collect
        ),
    )

    def execute(request, output):
        method = request["settings"]["method"]
        if method == "remeasure":
            calls["remeasure"] += 1
            assert set(request["settings"]["methods"]) == {"tiletune"}
            return dict(status="remeasured", validation=dict(tiletune=dict(samples_ms=[1] * 7)))
        assert method == "top_k"
        calls["tiletune"] += 1
        report = ranking(configs)
        report["selection"] = dict(requested_k=request["settings"]["top_k"], selected_indices=[0])
        write(output / "tiletune.json", report)
        write(output / "experiment.json", dict(workload=workload, configs=configs, measurement_identity=measurement))
        return dict(status="completed", correctness="passed", tuning_seconds=1, winner=dict(config=configs[0], index=0))

    monkeypatch.setattr(suite, "_existing_or_run", execute)
    baseline_root = tmp_path / "baselines"
    for revision in ("one", "two"):
        output = tmp_path / revision
        output.mkdir()
        study.execute(plan, output, settings, baseline_root=baseline_root)
        references = json.loads((output / "baselines.json").read_text())
        assert references[device.name]["gemm_fp8"]["reused"] == (revision == "two")
        current_hashes = baseline_store.hash_files(baseline_root.rglob("*.json"), baseline_root)
        if revision == "one":
            original_hashes = current_hashes
            plan = deepcopy(plan)
            plan["top_k"] = 1
        else:
            assert current_hashes == original_hashes
        curve = json.loads(next(output.rglob("oracle-curves.json")).read_text())
        carver = curve["methods"][0]
        if carver_status == "unsupported":
            assert carver["status"] == "unsupported"
        else:
            assert carver["status"] == "evaluated" and carver["available_count"] == 0
            assert all(row["status"] == "no_success" and row["oracle_at_k"] is None for row in carver["curves"])
            assert carver["saved_selection"]["selected_indices"] == []
        assert curve["methods"][2]["curves"][0]["oracle_at_k"] == 1
    assert calls == dict(collection=1, tiletune=2, remeasure=2)
    bundle = Path(references[device.name]["gemm_fp8"]["path"])
    artifact = bundle / "collection" / device.name / "test" / workload["name"] / "xgboost/xgboost.json"
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        baseline_store.verify_bundle(bundle, identity)


def test_result_comparison_accepts_ranking_revision_only_when_measurement_identity_matches(tmp_path):
    paths = [tmp_path / name / "report.json" for name in ("old", "new")]
    for i, path in enumerate(paths):
        write(path, {})
        write(
            path.parent / "experiment.json",
            dict(
                workload={"op": "gemm"},
                measurement_identity={"kernel": "same"},
                native_build=str(i),
                source_sha256={"tiletune/rank.py": str(i)},
            ),
        )
    assert "measurement identity" in check_metadata(*paths)
    write(paths[1].parent / "experiment.json", dict(workload={"op": "gemm"}, measurement_identity={"kernel": "different"}))
    with pytest.raises(ValueError, match="measurement identity differs"):
        check_metadata(*paths)


def test_monitor_kills_contended_worker_and_records_rejection(tmp_path, monkeypatch):
    gpu = dict(index="0", uuid="GPU-test", name="test", **{"utilization.gpu": "0"})
    observations = iter([
        dict(gpus=[gpu], processes=[]),
        dict(gpus=[gpu], processes=[dict(pid="999998", gpu_uuid="GPU-test")]),
        dict(gpus=[gpu], processes=[dict(pid=p, gpu_uuid="GPU-test") for p in ("999998", "999999")]),
    ])
    monkeypatch.setattr(monitor, "snapshot", lambda: next(observations))
    monkeypatch.setattr(monitor, "foreign_processes", lambda obs, *args: obs["processes"])
    monkeypatch.setattr(monitor.time, "sleep", lambda _: None)

    class Process:
        pid = 123456
        killed = False

        def poll(self):
            return -9 if self.killed else None

    process = Process()
    monkeypatch.setattr(monitor.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(monitor, "stop_worker", lambda p: setattr(p, "killed", True))
    with pytest.raises(RuntimeError, match="discard this invocation"):
        monitor.run_monitored(["worker"], tmp_path, [gpu])
    assert process.killed
    assert json.loads((tmp_path / "monitor.json").read_text())["status"] == "contended"


def test_unresolvable_gpu_pid_is_busy(monkeypatch):
    def missing(pid):
        raise ProcessLookupError(pid)

    monkeypatch.setattr(monitor.os, "getpgid", missing)
    row = dict(pid="999999", gpu_uuid="GPU-test")
    assert monitor.foreign_processes(dict(processes=[row]), "GPU-test") == [row]


@pytest.mark.parametrize("replacement", [False, True])
def test_monitor_tracks_host_pids_without_local_pid_mapping(tmp_path, monkeypatch, replacement):
    gpu = dict(uuid="GPU-test", **{"utilization.gpu": "0"})
    observations = iter([
        dict(gpus=[gpu], processes=[]),
        dict(gpus=[gpu], processes=[dict(pid="900000", gpu_uuid="GPU-test")]),
        dict(gpus=[gpu], processes=[dict(pid="900001" if replacement else "900000", gpu_uuid="GPU-test")]),
    ])
    monkeypatch.setattr(monitor, "snapshot", lambda: next(observations))
    monkeypatch.setattr(monitor.time, "sleep", lambda _: None)

    class Process:
        pid = 100
        returncode = 0
        polls = 0

        def poll(self):
            self.polls += 1
            return 0 if self.polls >= 2 else None

    monkeypatch.setattr(monitor.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(monitor, "stop_worker", lambda p: None)
    if replacement:
        with pytest.raises(RuntimeError, match="discard this invocation"):
            monitor.run_monitored(["worker"], tmp_path, [gpu])
    else:
        monitor.run_monitored(["worker"], tmp_path, [gpu])
    audit = json.loads((tmp_path / "monitor.json").read_text())
    assert audit["status"] == ("contended" if replacement else "uncontended")


def test_rejected_measurements_are_not_left_under_successful_report_names(tmp_path, monkeypatch):
    from experiments.common import run
    from experiments.common.spec import Workload

    def rejected(request, device, output, **kwargs):
        write(output / "monitor.json", dict(status="contended"))
        write(output / "outcomes.json", [dict(status="benchmarked", latency_ms=1)])
        raise RuntimeError("contended")

    monkeypatch.setattr(run, "run_external", rejected)
    request = run.make_request(
        Workload("s", "gemm_fp8", dict(m=64, n=128, k=128, transpose_b=True), dtype="float8_e4m3fn"),
        Device("hopper", TARGETS["hopper"]),
        dict(method="brute_force"),
    )
    output = tmp_path / "case"
    result = run.run_case(request, output)
    assert result["status"] == "failed"
    assert not (output / "outcomes.json").exists()
    assert (output / "discarded-outcomes.json").exists()
