"""Manifest contracts, worker isolation and numerical workload validation."""

from dataclasses import replace
import json
import subprocess
import sys

import pytest

from experiments.common.spec import Device, TARGETS, Workload, default_workloads, load_manifest, support_reason
from experiments.common.run import make_request, run_case, targets_match, validate_request, validate_result


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
    code = (
        "from experiments.common.spec import default_workloads; from experiments.common import run; import sys; "
        "assert 'torch' not in sys.modules; assert 'tilelang' not in sys.modules; "
        "assert len(default_workloads()) == 25"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_default_workloads_match_the_family_suites():
    from experiments.suite import core_cases

    assert default_workloads() == core_cases("final")
    assert default_workloads(smoke=True) == core_cases("development")


@pytest.mark.parametrize("op", ["rmsnorm", "reduce_sum", "elementwise", "softmax"])
def test_retired_operations_are_rejected(op):
    with pytest.raises(ValueError, match="Unknown operation"):
        Workload("retired", op, {})


@pytest.mark.parametrize("dtype", ["float8_e4m3fnuz", "float8_e5m2fnuz"])
def test_incompatible_fp8_encodings_are_rejected(dtype):
    with pytest.raises(ValueError, match="Unsupported workload dtype"):
        replace(default_workloads()[0], dtype=dtype)


def test_manifest_roundtrip_and_declared_unsupported_cases():
    workloads = default_workloads(smoke=True)
    devices = [Device(name, target) for name, target in TARGETS.items()]
    data = dict(version=1, workloads=[w.to_dict() for w in workloads], devices=[d.to_dict() for d in devices])
    assert load_manifest(data) == (devices, workloads)
    assert support_reason(replace(workloads[0], dtype="float32"), devices[0])
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
    from experiments.common.spec import configurations

    workload = default_workloads(True)[0]
    grid = [{"block_m": 128, "block_n": 256, "k_l1": 64}]
    device = Device("ascend910", TARGETS["ascend910"], configs={workload.name: grid})
    assert configurations(workload, device) == grid
    assert workload.parameters == dict(m=64, n=4096, k=4096, transpose_b=True)
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
    from experiments.common.run import run_native

    workload = replace(default_workloads(True)[0], dtype="bfloat16")
    device = Device("hopper", TARGETS["hopper"], profiles={"float16": "not-loaded.json"})
    request = make_request(workload, device, {**SETTINGS, "metric": "pipeline_time"})
    result = run_native(request, tmp_path)
    assert result["status"] == "analyzed"
    assert result["scored"] == 0
    assert "no primitive profile for dtype bfloat16" in result["model_unknown_reason"]


def test_small_signal_corruption_fails_correctness():
    import torch
    from experiments.common.kernels import make_case

    case = make_case(Workload("attention", "attention", dict(batch=1, heads=1, sequence=128, dim=16)))
    inputs = [x * 0.01 for x in case.inputs("cpu", torch.Generator().manual_seed(123))]
    expected = case.reference(*inputs)
    assert expected.max() < 0.02
    with pytest.raises(AssertionError):
        case.check([torch.zeros_like(expected)], [expected])
    case.check([expected], [expected])


@pytest.mark.parametrize("name", [w.name for w in default_workloads(True)])
def test_kernel_build_closures_obey_autotuner_contract(name):
    from experiments.common.kernels import make_case

    case = make_case(next(w for w in default_workloads(True) if w.name == name))
    assert all(isinstance(cell.cell_contents, (int, float, str, bool, type(None))) for cell in case.build.__closure__ or [])


def test_gpu_fp8_boundaries():
    import torch
    import tilelang
    from experiments.common.kernels import make_case
    from tilelang.tiletune import current_target

    if not torch.cuda.is_available():
        pytest.skip("CUDA or ROCm required")
    w = Workload("fp8", "gemm_fp8", dict(m=128, n=128, k=128, transpose_b=True), dtype="float8_e4m3fn")
    reason = support_reason(w, Device("test", current_target()))
    if reason:
        pytest.skip(reason)
    case = make_case(w)
    program = case.build(block_M=64, block_N=64, block_K=128, threads=128, num_stages=1)
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
