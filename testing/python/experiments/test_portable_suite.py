"""Manifest contracts, worker isolation and numerical workload validation."""

from dataclasses import replace
import json
import subprocess
import sys

import pytest

from experiments.portable.spec import Device, TARGETS, Workload, default_workloads, load_manifest, support_reason
from experiments.portable.run import make_request, run_case, targets_match, validate_request, validate_result


SETTINGS = dict(
    method="analyze",
    metric="traffic_waves",
    memory_regime="streaming",
    top_k=2,
    config_indices=[0],
    trace=False,
    seed=123,
    workers=2,
    warmup=2,
    rep=3,
    timeout=30,
    case_timeout=120,
)


def test_plan_does_not_import_gpu_runtime():
    code = "from experiments.portable.spec import default_workloads; from experiments.portable import run; import sys; assert 'torch' not in sys.modules; assert 'tilelang' not in sys.modules; assert len(default_workloads()) >= 17"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_manifest_roundtrip_and_declared_unsupported_cases():
    workloads = default_workloads(smoke=True)
    devices = [Device(name, target) for name, target in TARGETS.items()]
    data = dict(version=1, workloads=[w.to_dict() for w in workloads], devices=[d.to_dict() for d in devices])
    assert load_manifest(data) == (devices, workloads)
    assert support_reason(next(w for w in workloads if w.name == "gemm_fp8"), devices[0])
    assert support_reason(workloads[0], devices[-1])
    assert not support_reason(workloads[0], replace(devices[-1], worker=["/external/python", "worker.py"]))
    assert targets_match({"kind": "cuda", "arch": "sm_90"}, {"kind": "cuda", "arch": "sm_90a"})
    assert not targets_match(TARGETS["hopper"], TARGETS["blackwell"])


def test_request_and_result_cannot_substitute_a_different_experiment():
    w, d = default_workloads(True)[0], Device("hopper", TARGETS["hopper"])
    request = make_request(w, d, {**SETTINGS, "method": "exhaustive"})
    assert validate_request(request) == (w, d)
    modified = {**request, "settings": {**SETTINGS, "seed": 4}}
    with pytest.raises(ValueError, match="hash"):
        validate_request(modified)
    result = dict(
        version=1,
        request_id=request["request_id"],
        workload=w.name,
        device=d.name,
        status="completed",
        winner=dict(latency_ms=1),
        correctness="passed",
        device_observation=dict(name="test", target=TARGETS["blackwell"]),
    )
    with pytest.raises(ValueError, match="different target"):
        validate_result(result, request)
    result["device_observation"]["target"] = TARGETS["hopper"]
    assert validate_result(result, request) == result
    result["workload"] = "another_kernel"
    with pytest.raises(ValueError, match="identity"):
        validate_result(result, request)


def test_failed_external_worker_keeps_case_result(tmp_path):
    device = Device("ascend910", TARGETS["ascend910"], worker=[sys.executable, "-c", "raise SystemExit(7)"])
    request = make_request(default_workloads(True)[0], device, SETTINGS)
    result = run_case(request, tmp_path / "failed")
    assert result["status"] == "failed"
    assert "code 7" in result["reason"]
    assert json.loads((tmp_path / "failed/result.json").read_text()) == result


def test_target_grids_preserve_workload_semantics():
    from experiments.portable.spec import configurations

    workload = default_workloads(True)[0]
    grid = [{"block_m": 128, "block_n": 256, "k_l1": 64}]
    device = Device("ascend910", TARGETS["ascend910"], configs={workload.name: grid})
    assert configurations(workload, device) == grid
    assert workload.parameters == dict(m=128, n=128, k=128)
    with pytest.raises(ValueError, match="unknown workloads"):
        load_manifest(dict(version=1, workloads=[workload.to_dict()], devices=[replace(device, configs={"typo": grid}).to_dict()]))
    with pytest.raises(ValueError, match="input dtypes"):
        Device("hopper", TARGETS["hopper"], profiles={"fp16_typo": "profile.json"})


def test_case_timeout_keeps_failure_record(tmp_path):
    device = Device("ascend910", TARGETS["ascend910"], worker=[sys.executable, "-c", "import time; time.sleep(60)"])
    request = make_request(default_workloads(True)[0], device, {**SETTINGS, "case_timeout": 0.1})
    result = run_case(request, tmp_path / "timeout")
    assert result["status"] == "failed"
    assert "TimeoutExpired" in result["reason"]


def test_missing_dtype_profile_keeps_analysis_available(tmp_path):
    from experiments.portable.run import run_native

    workload = replace(default_workloads(True)[0], dtype="bfloat16")
    device = Device("hopper", TARGETS["hopper"], profiles={"float16": "not-loaded.json"})
    request = make_request(workload, device, {**SETTINGS, "metric": "pipeline_time"})
    result = run_native(request, tmp_path)
    assert result["status"] == "analyzed"
    assert result["scored"] == 0
    assert "no primitive profile for dtype bfloat16" in result["model_unknown_reason"]


def test_analysis_accepts_per_candidate_pass_configs(tmp_path):
    from experiments.portable.run import run_native

    workload = Workload("softmax", "softmax", dict(rows=4, columns=128))
    grid = [{"block_rows": 1, "threads": 128, "pass_configs": {"tl.enable_fast_math": True}}]
    device = Device("hopper", TARGETS["hopper"], configs={workload.name: grid})
    result = run_native(make_request(workload, device, SETTINGS), tmp_path)
    assert result["status"] == "analyzed", result
    report = json.loads((tmp_path / "tiletune.json").read_text())
    assert report["configs"][0]["config"] == grid[0]


def test_small_signal_corruption_fails_correctness():
    import torch
    from experiments.portable.kernels import make_case

    case = make_case(Workload("softmax", "softmax", dict(rows=2, columns=4096)))
    inputs = case.inputs("cpu", torch.Generator().manual_seed(123))
    expected = case.reference(*inputs)
    assert expected.max() < 0.02
    with pytest.raises(AssertionError):
        case.check([torch.zeros_like(expected)], [expected])
    case.check([expected], [expected])


@pytest.mark.parametrize(
    "name", ["gemm_nn", "gemm_nt", "gemm_batched", "gemm_bias_relu", "flashattention", "kda_recurrent", "kda_chunk_o", "softmax"]
)
def test_kernel_build_closures_obey_autotuner_contract(name):
    from experiments.portable.kernels import make_case

    case = make_case(next(w for w in default_workloads(True) if w.name == name))
    assert all(isinstance(cell.cell_contents, (int, float, str, bool, type(None))) for cell in case.build.__closure__ or [])


@pytest.mark.parametrize("op", ["softmax", "rmsnorm", "reduce_sum", "elementwise"])
def test_gpu_row_kernel_boundaries(op):
    import torch
    import tilelang
    from experiments.portable.kernels import make_case
    from tilelang.tiletune import current_target

    if not torch.cuda.is_available():
        pytest.skip("CUDA or ROCm required")
    w = Workload(op, op, dict(rows=7, columns=93))
    case = make_case(w)
    program = case.build(block_rows=2, threads=128)
    kernel = tilelang.compile(program, target=current_target(), execution_backend="tvm_ffi", out_idx=case.out_idx)
    inputs = case.inputs("cuda", torch.Generator(device="cuda").manual_seed(123))
    result = kernel(*inputs)
    case.check([result], [case.reference(*inputs)])


def test_native_worker_analyzes_cross_target_without_hardware(tmp_path):
    # This exercises the complete native worker protocol on a cross target.
    device = Device("mi308", TARGETS["mi308"])
    request = make_request(default_workloads(True)[0], device, SETTINGS)
    result = run_case(request, tmp_path / "analysis")
    assert result["status"] == "analyzed", result
    assert result["configs"] == 1
    assert "device_observation" not in result
    assert (tmp_path / "analysis/tiletune.json").exists()
