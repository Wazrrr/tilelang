"""Learned selection, workload separation, provenance and failure accounting."""

from copy import deepcopy
import hashlib
import importlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments.portable.spec import Device, TARGETS, Workload
from experiments.portable.run import make_request, validate_request, validate_result
from experiments.xgboost.data import canonical_workload, features, make_context, read_runs, workload_key
from experiments.xgboost.model import Predictor, evaluate, train


GRID = [{"block_rows": b, "threads": t} for b in (1, 2, 4) for t in (128, 256)]


def write_run(path, rows, *, name="softmax", failed=()):
    path.mkdir(parents=True)
    workload = Workload(name, "softmax", dict(rows=rows, columns=128))
    experiment = dict(
        workload=workload.to_dict(),
        device=Device("test", TARGETS["hopper"]).to_dict(),
        settings=dict(method="exhaustive"),
        configs=GRID,
        device_observation=dict(name="Test GPU", target=TARGETS["hopper"]),
        source_sha256={"experiments/portable/kernels.py": "same-kernel-source"},
        native_build="test-compiler-build",
    )
    records = [
        dict(
            index=i,
            config=config,
            status="compilation_failed" if i in failed else "benchmarked",
            latency_ms=None if i in failed else 4 / config["block_rows"] + 0.1 * (config["threads"] == 256) + rows / 10000,
        )
        for i, config in enumerate(GRID)
    ]
    (path / "experiment.json").write_text(json.dumps(experiment))
    (path / "tiletune.json").write_text(json.dumps(dict(configs=records)))
    (path / "result.json").write_text(json.dumps(dict(status="completed", correctness="passed", tuning_seconds=1.0)))
    return path


@pytest.fixture
def trained(tmp_path):
    pytest.importorskip("xgboost")
    training = read_runs([write_run(tmp_path / "train", 16)])
    validation = read_runs([write_run(tmp_path / "validation", 32)])
    heldout = read_runs([write_run(tmp_path / "test", 64)])
    path = tmp_path / "model.json"
    train(training, validation, path, rounds=40, learning_rate=0.2, max_depth=3, workers=1)
    return path, training, validation, heldout


def test_optional_imports_do_not_load_accelerator_or_ml_runtime():
    code = "import experiments.xgboost.model, experiments.xgboost.data, experiments.xgboost.integration; import sys; assert not {'xgboost', 'torch', 'tilelang', 'numpy'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_model_learns_and_roundtrips_without_test_labels(trained):
    path, training, validation, heldout = trained
    predictor = Predictor(path, workers=1)
    context = heldout[0]["context"]
    report = predictor.rank(context, GRID, 2)
    assert report["selection"]["selected_indices"] == [4, 5]
    assert report["selection"]["selected_count"] == 2
    reversed_keys = [dict(reversed(list(config.items()))) for config in GRID]
    assert Predictor(path, workers=1).predict(context, reversed_keys) == predictor.predict(context, GRID)
    for runs in (training, validation):
        with pytest.raises(ValueError, match="used for training or validation"):
            predictor.rank(runs[0]["context"], GRID, 2)
    with pytest.raises(FileExistsError):
        train(training, validation, path)


def test_split_is_semantic_not_run_name_or_configuration_subset(tmp_path):
    pytest.importorskip("xgboost")
    training = read_runs([write_run(tmp_path / "train", 16)])
    validation = read_runs([write_run(tmp_path / "alias", 16, name="renamed_softmax")])
    validation[0]["samples"] = validation[0]["samples"][3:]
    with pytest.raises(ValueError, match="workload overlap"):
        train(training, validation, tmp_path / "bad.json")
    first = make_context(
        Workload("a", "gemm", dict(m=32, n=64, k=128)),
        "gemm",
        TARGETS["hopper"],
        "Test GPU",
        "event",
        {"experiments/gemm/kernel.py": "same"},
    )
    second = make_context(
        Workload("alias", "gemm", dict(m=32, n=64, k=128, batch=1, transpose_a=False)),
        "gemm",
        TARGETS["hopper"],
        "Test GPU",
        "event",
        {"experiments/gemm/kernel.py": "same"},
    )
    assert workload_key(first) == workload_key(second)
    a = dict(workload=canonical_workload(Workload("a", "rmsnorm", dict(rows=4, columns=8, epsilon=1))))
    b = dict(workload=canonical_workload(Workload("b", "rmsnorm", dict(rows=4, columns=8, epsilon=1.0))))
    assert workload_key(a) == workload_key(b)


def test_duplicate_configurations_cannot_ambiguously_identify_the_winner(trained):
    path, _, _, heldout = trained
    with pytest.raises(ValueError, match="unique configurations"):
        Predictor(path, workers=1).rank(heldout[0]["context"], [GRID[0], dict(reversed(list(GRID[0].items())))], 2)


@pytest.mark.parametrize("change", ["device", "source", "backend", "compiler"])
def test_model_cannot_silently_change_execution_domain(trained, change):
    path, _, _, heldout = trained
    context = deepcopy(heldout[0]["context"])
    if change == "device":
        context["device"]["name"] = "Another GPU"
    elif change == "source":
        context["kernel_sha256"]["experiments/portable/kernels.py"] = "changed-kernel"
    elif change == "backend":
        context["benchmark_backend"] = "cudagraph"
    else:
        context["environment"]["native_build"] = "rebuilt compiler"
    with pytest.raises(ValueError, match="no matching"):
        Predictor(path, workers=1).rank(context, GRID, 2)


def test_model_fingerprint_is_frozen_in_worker_request(trained):
    path, _, _, _ = trained
    workload = Workload("test", "softmax", dict(rows=64, columns=128))
    request = make_request(workload, Device("hopper", TARGETS["hopper"]), dict(method="xgboost", xgb_model=str(path)))
    assert validate_request(request)[0] == workload
    expected = request["settings"]["xgb_model_sha256"]
    assert expected == hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after"):
        Predictor(path, expected_sha256=expected)


def test_worker_result_must_keep_model_and_selection_budget(trained):
    path, _, _, _ = trained
    request = make_request(
        Workload("test", "softmax", dict(rows=64, columns=128)),
        Device("hopper", TARGETS["hopper"]),
        dict(method="xgboost", xgb_model=str(path), top_k=2),
    )
    result = dict(
        version=1,
        request_id=request["request_id"],
        workload="test",
        device="hopper",
        status="completed",
        correctness="passed",
        device_observation=dict(name="Test GPU", target=TARGETS["hopper"]),
        model_sha256=request["settings"]["xgb_model_sha256"],
        winner=dict(index=1, latency_ms=1),
        selection=dict(requested_k=2, selected_count=2, selected_indices=[0, 1]),
    )
    assert validate_result(result, request) == result
    result["model_sha256"] = "different"
    with pytest.raises(ValueError, match="different XGBoost model"):
        validate_result(result, request)
    result["model_sha256"] = request["settings"]["xgb_model_sha256"]
    result["selection"].update(selected_indices=[0, 1, 2], selected_count=3)
    with pytest.raises(ValueError, match="selection budget"):
        validate_result(result, request)


def test_training_requires_compiler_identity_and_correctness(tmp_path):
    path = write_run(tmp_path / "run", 16)
    metadata = json.loads((path / "experiment.json").read_text())
    del metadata["native_build"]
    (path / "experiment.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="compiler build fingerprint"):
        read_runs([path])
    metadata["native_build"] = "build"
    (path / "experiment.json").write_text(json.dumps(metadata))
    (path / "result.json").write_text(json.dumps(dict(status="completed")))
    with pytest.raises(ValueError, match="correctness"):
        read_runs([path])


def test_features_exclude_measurements_and_analytical_scores(trained):
    _, _, _, heldout = trained
    context = deepcopy(heldout[0]["context"])
    before = features(context, GRID[0])
    context.update(latency_ms=123, predicted_rank=2, pressure={"registers": 45}, tile_cost={"score": 99})
    assert features(context, GRID[0]) == before
    context["benchmark_backend"] = "cudagraph"
    assert features(context, GRID[0]) != before


def test_selected_elaboration_failures_leave_the_remaining_budget_intact():
    import tilelang  # Initialize the repository's TVM before importing its Target.
    from experiments.xgboost.execution import prepare_selected

    tried = []

    def builder(block):
        tried.append(block)
        if block == 0:
            raise ValueError("bad selected tile")
        return tilelang

    configs = [dict(block=0), dict(block=64), dict(block=128)]
    valid, failures = prepare_selected(builder, configs, [0, 1], TARGETS["hopper"])
    assert valid == [1] and tried == [0, 64]
    assert failures[0]["status"] == "elaboration_failed"
    assert "bad selected tile" in failures[0]["error"]


def _build_with_bad_rows(block_rows, threads):
    from experiments.portable.kernels import make_case

    if block_rows <= 0:
        raise ValueError("bad selected tile")
    return make_case(Workload("softmax", "softmax", dict(rows=16, columns=128))).build(block_rows, threads)


@pytest.mark.parametrize("all_failed", [False, True])
def test_gpu_frozen_selection_keeps_elaboration_failures_and_original_indices(tmp_path, monkeypatch, all_failed):
    import torch
    from experiments.portable.kernels import make_case
    from experiments.xgboost.execution import run_selected
    from tilelang.tiletune import current_target

    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm required")
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")
    monkeypatch.setenv("TILELANG_AUTO_TUNING_DISABLE_CACHE", "1")
    monkeypatch.setenv("TILELANG_AUTO_TUNING_CPU_COUNTS", "2")
    case = make_case(Workload("softmax", "softmax", dict(rows=16, columns=128)))
    case.build = _build_with_bad_rows
    configs = [dict(block_rows=b, threads=128) for b in [0, -1 if all_failed else 1, 2]]
    report = dict(
        metric="xgboost_log_latency",
        score_units="log(ms)",
        model_sha256="test",
        selection=dict(requested_k=2, selected_indices=[0, 1], selected_count=2, wall_time_ms=0),
        ranking=[dict(index=i, rank=i + 1) for i in range(3)],
        configs=[dict(index=i, config=c, selected=i < 2, status="selected" if i < 2 else "not_selected") for i, c in enumerate(configs)],
    )
    inputs = case.inputs("cuda", torch.Generator(device="cuda").manual_seed(123))
    result = run_selected(
        case, configs, [7, 11, 30], current_target(), inputs, case.reference(*inputs), dict(warmup=2, rep=3, timeout=30), tmp_path, report
    )
    saved = json.loads((tmp_path / "xgboost.json").read_text())
    assert saved["selection"]["selected_indices"] == [0, 1]
    assert saved["configs"][0]["status"] == "elaboration_failed"
    assert saved["configs"][2]["status"] == "not_selected"
    if all_failed:
        assert result["status"] == "failed"
        assert saved["configs"][1]["status"] == "elaboration_failed"
    else:
        assert result["status"] == "completed", result
        assert result["winner"]["original_index"] == 11
        assert saved["configs"][1]["status"] == "benchmarked"


@pytest.mark.parametrize("latency", [0, -1, True, float("nan"), float("inf")])
def test_invalid_measurements_are_not_training_labels(tmp_path, latency):
    path = write_run(tmp_path / "run", 16)
    report = json.loads((path / "tiletune.json").read_text())
    report["configs"][0]["latency_ms"] = latency
    (path / "tiletune.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="invalid successful latency"):
        read_runs([path])


def test_exhaustive_failures_are_retained_and_do_not_get_zero_labels(tmp_path):
    path = write_run(tmp_path / "run", 16, failed=(0, 3))
    run = read_runs([path])[0]
    assert len(run["configs"]) == 6 and len(run["samples"]) == 4
    assert run["provenance"]["statuses"]["compilation_failed"] == 2
    metadata = json.loads((path / "experiment.json").read_text())
    metadata["settings"]["method"] = "top_k"
    (path / "experiment.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="expected an exhaustive run"):
        read_runs([path])


@pytest.mark.parametrize("status", ["pending", "compiled", "not_attempted", "not_selected"])
def test_incomplete_exhaustive_runs_cannot_become_oracles(tmp_path, status):
    path = write_run(tmp_path / "run", 16)
    records = json.loads((path / "tiletune.json").read_text())
    records["configs"][1]["status"] = status
    (path / "tiletune.json").write_text(json.dumps(records))
    with pytest.raises(ValueError, match="incomplete candidate"):
        read_runs([path])
    (path / "result.json").unlink()
    with pytest.raises(ValueError, match="missing completed"):
        read_runs([path])


def test_oracle_does_not_replace_failed_selected_candidates(trained):
    path, _, _, heldout = trained
    predictor = Predictor(path, workers=1)
    run = heldout[0]
    run["samples"] = [sample for sample in run["samples"] if sample["index"] != 4]
    report = evaluate(predictor, [run], 2)["results"][0]
    assert report["selection"]["selected_indices"] == [4, 5]
    assert report["selected_candidates_with_oracle_measurement"] == 1
    assert report["oracle_at_k"] == 1


def test_repeated_runs_are_aggregated_before_training(tmp_path):
    pytest.importorskip("xgboost")
    training = read_runs([write_run(tmp_path / "train1", 16), write_run(tmp_path / "train2", 16)])
    validation = read_runs([write_run(tmp_path / "validation", 32)])
    result = train(training, validation, tmp_path / "model.json", rounds=2, workers=1)
    assert result["training_samples"] == len(GRID)
    assert len(result["training_runs"]) == 2


@pytest.mark.parametrize("family", ["gemm_fp8", "flash_attention"])
def test_comparison_children_include_xgboost_and_keep_fingerprint(tmp_path, monkeypatch, family):
    from experiments._common import prepare_run, write_json

    runner = importlib.import_module(f"experiments.{family}.tiletune.run")
    model = tmp_path / "model.json"
    model.write_text("model placeholder; child execution is mocked")
    args = runner.parse_args(["--method", "all", "--xgb-model", str(model), "--top-k", "2", "--output", str(tmp_path / "runs")])
    prepare_run(args)
    children = []

    def run_child(command, *, stdout, stderr):
        child = runner.parse_args(command[3:])
        assert child.xgb_model == model and child.xgb_model_sha256 == args.xgb_model_sha256
        prepare_run(child)
        write_json(child.output / "summary.json", dict(method=child.method, status="failed"))
        children.append(child.method)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(runner.subprocess, "run", run_child)
    monkeypatch.setattr(runner, "remeasure_winners", lambda *args: None)
    assert not runner.run_all(args)
    assert children == ["tiletune", "xgboost", "brute_force"]
