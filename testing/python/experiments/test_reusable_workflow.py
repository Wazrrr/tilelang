"""Baseline reuse, system ablations and contention rejection without a GPU."""

from copy import deepcopy
from contextlib import nullcontext
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


def test_mixed_dtype_profile_preparation_covers_each_supported_dtype_once(tmp_path):
    from dataclasses import replace
    from experiments.common.study import prepare_profiles
    from experiments.common.spec import default_workloads, Device, TARGETS

    calls = []

    def profile(request, output):
        dtype = request["workload"]["dtype"]
        calls.append(dtype)
        assert output.name == dtype
        return dict(status="profiled", profiles={dtype: str(output / "primitive-profile.json")})

    workloads = [w.to_dict() for w in default_workloads()]
    device = Device("hopper", TARGETS["hopper"])
    prepared = prepare_profiles(device, workloads, tmp_path, {}, profile)
    assert calls == ["bfloat16", "float8_e4m3fn"]
    assert set(prepared.profiles) == set(calls)
    assert prepare_profiles(prepared, workloads, tmp_path, {}, profile) is prepared
    calls.clear()
    prepared = prepare_profiles(replace(device, name="ampere", target=TARGETS["ampere"]), workloads, tmp_path, {}, profile)
    assert calls == ["bfloat16"] and set(prepared.profiles) == {"bfloat16"}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, data)


@pytest.mark.parametrize(
    "family,count", [("gemm", 3456), ("flash_attention", 576), ("kda", 1296), ("gemm_fp8", 576), ("grouped_gemm", 576)]
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


def test_baseline_identity_ignores_tiletune_seed_and_k_but_includes_baseline_seed(monkeypatch):
    plan, device, _ = fixture_plan()
    settings = dict(warmup=1, rep=2, timeout=30, case_timeout=300, workers=1)
    monkeypatch.setattr(baseline_store, "measurement_sources", lambda families: {"kernel": "hash"})
    _, first = baseline_store.identities(plan, device, settings, {"device": "test"})
    for family in ("gemm", "flash_attention", "gemm_fp8", "kda", "grouped_gemm"):
        assert f"experiments/{family}/carver.py" in first["sources"]
    assert "experiments/common/carver.py" in first["sources"]
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
    identity = dict(splits=plan["splits"], pools={workload["name"]: configs}, baseline_seed=123, measurement=measurement)
    settings = dict(warmup=1, rep=2, timeout=30, case_timeout=300, workers=1)
    calls = dict(collection=0, tiletune=0, remeasure=0)
    monkeypatch.setattr(study, "runtime_identity", lambda device: {})
    monkeypatch.setattr(study, "cuda_device", lambda device: nullcontext())
    monkeypatch.setattr(study, "identities", lambda *args: (measurement, identity))

    def collect(command, *, check):
        calls["collection"] += 1
        assert check and command[command.index("--methods") + 1 : command.index("--methods") + 3] == ["carver", "xgboost"]
        base = Path(command[command.index("--output") + 1]) / device.name / "test" / workload["name"]
        report = ranking(configs)
        success = dict(
            status="completed",
            correctness="passed",
            tuning_seconds=10,
            winner=dict(config=configs[0], index=0),
            selection=report["selection"],
        )
        methods = dict(brute_force=success, xgboost=success, carver=dict(status=carver_status, reason="no eligible Carver candidates"))
        write(base / "methods.json", methods)
        for method, result in methods.items():
            write(base / method / "result.json", result)
            if method != "carver":
                write(base / method / (method + ".json"), report)
                write(base / method / "experiment.json", dict(workload=workload, configs=configs, measurement_identity=measurement))
        if carver_status == "model_unavailable":
            rejected = deepcopy(report)
            rejected["selection"]["selected_indices"] = []
            for row in rejected["configs"]:
                row.update(status="model_rejected")
                del row["latency_ms"]
            for row in rejected["ranking"]:
                row.update(tier="unknown", score=None)
            write(base / "carver/carver.json", rejected)
        write(base / "brute_force/outcomes.json", report["configs"])

    monkeypatch.setattr(
        study,
        "collect_bundle",
        lambda root, identity, measurement, device, settings, **options: baseline_store.collect_bundle(
            root, identity, measurement, device, settings, run=collect, **options
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
    collection_output = tmp_path / "explicit-collection"
    collection_output.mkdir()
    assert study.execute(plan, collection_output, settings, baseline_root=baseline_root, run_baselines=True) == 0
    assert calls == dict(collection=1, tiletune=0, remeasure=0)
    for revision in ("one", "two"):
        output = tmp_path / revision
        output.mkdir()
        study.execute(plan, output, settings, baseline_root=baseline_root)
        references = json.loads((output / "baselines.json").read_text())
        assert references[device.name]["gemm_fp8"]["reused"]
        assert references[device.name]["gemm_fp8"]["seed"] == 123
        current_hashes = baseline_store.hash_files(baseline_root.rglob("*.json"), baseline_root)
        if revision == "one":
            original_hashes = current_hashes
            plan = deepcopy(plan)
            plan["top_k"] = 1
            measurement["compiler"] = "updated"
            identity["baseline_seed"] = 456
            identity["sources"] = {"baseline_store.py": "new code"}
            settings["rep"] = 3
        else:
            assert current_hashes == original_hashes
        curve = json.loads(next(output.rglob("oracle-curves.json")).read_text())
        if carver_status == "unsupported":
            assert curve["methods"][0]["status"] == "unsupported"
        else:
            assert curve["methods"][0]["available_count"] == 0
            assert all(row["oracle_at_k"] is None and row["shortfall"] == row["k"] for row in curve["methods"][0]["curves"])
        assert curve["methods"][2]["curves"][0]["oracle_at_k"] == 1
    assert calls == dict(collection=1, tiletune=2, remeasure=2)
    bundle = Path(references[device.name]["gemm_fp8"]["path"])
    old_bundle_hashes = baseline_store.hash_files(bundle.rglob("*.json"), bundle)
    refresh_output = tmp_path / "explicit-refresh"
    refresh_output.mkdir()
    assert study.execute(plan, refresh_output, settings, baseline_root=baseline_root, run_baselines=True) == 0
    refreshed = json.loads((refresh_output / "baselines.json").read_text())[device.name]["gemm_fp8"]
    assert not refreshed["reused"] and Path(refreshed["path"]) != bundle
    assert calls == dict(collection=2, tiletune=2, remeasure=2)
    assert baseline_store.hash_files(bundle.rglob("*.json"), bundle) == old_bundle_hashes
    # Resuming a study must keep its original measurements after a refresh.
    pinned = tmp_path / "two/baselines.json"
    before = (pinned.read_bytes(), pinned.stat().st_mtime_ns)
    study.execute(plan, tmp_path / "two", settings, baseline_root=baseline_root)
    assert (pinned.read_bytes(), pinned.stat().st_mtime_ns) == before
    assert json.loads(pinned.read_text())[device.name]["gemm_fp8"]["path"] == str(bundle)
    # A new study instead uses the newly published bundle.
    latest = tmp_path / "three"
    latest.mkdir()
    study.execute(plan, latest, settings, baseline_root=baseline_root)
    assert json.loads((latest / "baselines.json").read_text())[device.name]["gemm_fp8"]["path"] == refreshed["path"]
    assert calls == dict(collection=2, tiletune=4, remeasure=4)
    artifact = bundle / "collection" / device.name / "test" / workload["name"] / "xgboost/xgboost.json"
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        baseline_store.load_bundle(baseline_root, identity, reference=references[device.name]["gemm_fp8"])


def test_missing_or_changed_baselines_are_read_only_and_failed_refresh_keeps_current(tmp_path):
    root = tmp_path / "saved"
    plan, device, configs = fixture_plan()
    workload = plan["splits"]["test"][0]
    identity = dict(splits=plan["splits"], pools={workload["name"]: configs}, baseline_seed=123)
    with pytest.raises(FileNotFoundError, match="--run-baselines"):
        baseline_store.load_bundle(root, identity)
    assert not root.exists()
    bundle = root / "runs/previous"
    write(bundle / "complete.json", dict(identity=identity, artifacts={}))
    baseline_store.publish_bundle(root, bundle, identity)
    current = root / "current.json"
    before = (current.read_bytes(), current.stat().st_mtime_ns)
    assert baseline_store.load_bundle(root, identity) == (bundle, True)
    changed = deepcopy(identity)
    changed["pools"][workload["name"]] = configs[:1]
    with pytest.raises(ValueError, match="explicitly rerun"):
        baseline_store.load_bundle(root, changed)
    assert (current.read_bytes(), current.stat().st_mtime_ns) == before

    def fail(*args, **kwargs):
        raise RuntimeError("collection interrupted")

    with pytest.raises(RuntimeError, match="collection interrupted"):
        baseline_store.collect_bundle(root, identity, {}, device, {}, refresh=True, run=fail)
    assert (current.read_bytes(), current.stat().st_mtime_ns) == before
    assert baseline_store.load_bundle(root, identity) == (bundle, True)


@pytest.mark.parametrize("change", ["provenance", "order", "training", "pool", "workload", "gpu", "contract"])
def test_baseline_reuse_depends_only_on_requested_case_gpu_and_pool(tmp_path, change):
    plan, _, configs = fixture_plan()
    workload = plan["splits"]["test"][0]
    identity = dict(
        splits=plan["splits"],
        pools={workload["name"]: configs},
        measurement=dict(runtime=dict(device="H200", target=TARGETS["hopper"], driver="old")),
        baseline_seed=123,
    )
    bundle = tmp_path / "runs/original"
    write(bundle / "complete.json", dict(identity=identity, artifacts={}))
    baseline_store.publish_bundle(tmp_path, bundle, identity)
    changed = deepcopy(identity)
    if change == "provenance":
        changed["measurement"]["runtime"]["driver"] = "new"
        changed["measurement"]["sources"] = {"kernel.py": "new"}
        changed["measurement"]["timing"] = {"rep": 500}
        changed["baseline_seed"] = 456
        changed["xgboost"] = {"max_depth": 8}
    elif change == "order":
        changed["pools"][workload["name"]].reverse()
    elif change == "training":
        changed["splits"]["train"] = []
        changed["splits"]["validation"] = []
    elif change == "contract":
        changed["measurement"]["kernel_contract_version"] = 2
    elif change == "pool":
        changed["pools"][workload["name"]] = configs[:1]
    elif change == "workload":
        changed["splits"]["test"][0]["parameters"]["m"] += 1
    else:
        changed["measurement"]["runtime"]["device"] = "H100"
    if change in ("pool", "workload", "gpu", "contract"):
        with pytest.raises(ValueError, match="explicitly rerun"):
            baseline_store.load_bundle(tmp_path, changed)
    else:
        assert baseline_store.load_bundle(tmp_path, changed) == (bundle, True)


@pytest.mark.parametrize("family", ["gemm", "flash_attention", "kda", "gemm_fp8", "grouped_gemm"])
def test_default_baseline_location_is_family_and_gpu_specific(family):
    device = Device("hopper", TARGETS["hopper"])
    path = baseline_store.storage_root(family, device, dict(device="NVIDIA H200"))
    assert path == baseline_store.ROOT / "experiments" / family / "results/H200/baselines"


def test_unavailable_baseline_target_cannot_report_success(tmp_path):
    device = Device("ascend910b", TARGETS["ascend910b"])
    plan = study_plan("full", [device], families=["gemm"])
    assert study.execute(plan, tmp_path, {}, run_baselines=True) == 1
    result = json.loads((tmp_path / "baseline-collection.json").read_text())
    assert result["status"] == "incomplete"
    assert result["unavailable"] and not result["baselines"]


@pytest.mark.parametrize("corruption", ["oracle_pool", "method_pool", "duplicate_rank", "unknown_rank", "winner", "selection"])
def test_baseline_publication_rejects_wrong_pools_and_inconsistent_rankings(tmp_path, corruption):
    _, _, configs = fixture_plan()
    report = ranking(configs)
    result = dict(status="completed", selection=deepcopy(report["selection"]), winner=dict(index=0, config=configs[0]))
    oracle = deepcopy(report["configs"])
    if corruption == "oracle_pool":
        oracle[0]["config"] = dict(different=1)
    elif corruption == "method_pool":
        report["configs"][0]["config"] = dict(different=1)
    elif corruption == "duplicate_rank":
        report["ranking"][1]["index"] = 0
    elif corruption == "unknown_rank":
        report["ranking"][1]["index"] = 100
    elif corruption == "winner":
        result["winner"]["config"] = configs[1]
    else:
        result["selection"]["selected_indices"] = [1]
    write(tmp_path / "brute_force/outcomes.json", oracle)
    write(tmp_path / "carver/result.json", dict(status="unsupported", reason="no adapter"))
    write(tmp_path / "xgboost/xgboost.json", report)
    write(tmp_path / "xgboost/result.json", result)
    with pytest.raises(ValueError):
        baseline_store.validate_case_bundle(tmp_path, configs)


def test_generated_family_results_do_not_change_source_identity(tmp_path, monkeypatch):
    from experiments.utils import cli

    monkeypatch.setattr(cli, "__file__", str(tmp_path / "experiments/utils/cli.py"))
    monkeypatch.setattr(cli, "SOURCE_ROOTS", ("experiments/gemm",))
    source = tmp_path / "experiments/gemm/kernel.py"
    source.parent.mkdir(parents=True)
    source.write_text("kernel")
    before = cli.source_hashes("experiments/gemm/kernel.py")
    generated = source.parent / "results/H200/recorded_worker.py"
    generated.parent.mkdir(parents=True)
    generated.write_text("recorded script")
    assert cli.source_hashes("experiments/gemm/kernel.py") == before


def test_result_comparison_records_provenance_changes_without_rejecting_saved_baselines(tmp_path):
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
    assert "provenance differs" in check_metadata(*paths)
    write(paths[1].parent / "experiment.json", dict(workload={"op": "gemm"}, measurement_identity={"kernel": "different"}))
    assert "measurement_identity" in check_metadata(*paths)


def test_monitor_kills_contended_worker_and_records_rejection(tmp_path, monkeypatch):
    gpu = dict(index="0", uuid="GPU-test", name="test", **{"utilization.gpu": "0"})
    observations = iter([dict(gpus=[gpu], processes=[]), dict(gpus=[gpu], processes=[dict(pid="999999", gpu_uuid="GPU-test")])])
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


def test_rejected_measurements_are_not_left_under_successful_report_names(tmp_path, monkeypatch):
    from experiments.common import run
    from experiments.common.spec import Workload

    def rejected(request, device, output, **kwargs):
        write(output / "monitor.json", dict(status="contended"))
        write(output / "outcomes.json", [dict(status="benchmarked", latency_ms=1)])
        raise RuntimeError("contended")

    monkeypatch.setattr(run, "run_external", rejected)
    request = run.make_request(
        Workload("s", "gemm_fp8", dict(m=32, n=128, k=128), dtype="float8_e4m3fn"),
        Device("hopper", TARGETS["hopper"]),
        dict(method="brute_force"),
    )
    output = tmp_path / "case"
    result = run.run_case(request, output)
    assert result["status"] == "failed"
    assert not (output / "outcomes.json").exists()
    assert (output / "discarded-outcomes.json").exists()
