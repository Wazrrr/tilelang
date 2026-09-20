"""One experiment definition across CUDA architectures, without GPU execution."""

import os

import pytest

from experiments.backend import FP8_COMPUTE_DTYPE
from experiments.common.baselines import carver_support_reason
from experiments.common.run import make_request, run_native
from experiments.common.smoke import instruction_evidence
from experiments.common.spec import Device, TARGETS, configuration_space, support_reason
from experiments.families import FAMILIES, family_module
from experiments.suite import study_plan
from experiments.utils import monitor


@pytest.mark.parametrize("op", FAMILIES)
def test_all_cuda_targets_preserve_final_cases_and_complete_pools(op):
    workloads = family_module(op, "cases").cases(holdout=True)
    original = None
    for name in ("ampere", "hopper", "blackwell"):
        device = Device(name, TARGETS[name])
        spaces = [configuration_space(w, device) for w in workloads]
        if op == "gemm_fp8" and name == "ampere" and FP8_COMPUTE_DTYPE == "float8_e4m3fn":
            assert all("FP8" in support_reason(w, device) for w in workloads)
        else:
            assert all(support_reason(w, device) is None for w in workloads)
        if original is None:
            original = spaces
        assert spaces == original
        plan = study_plan("full", [device], families=[FAMILIES[op]])
        assert not plan["unavailable"]
        for workload, space in zip(workloads, spaces):
            assert plan["subsets"][name][workload.name]["indices"] == list(range(len(space["configs"])))


@pytest.mark.parametrize("op", ["gemm", "grouped_gemm", "attention"])
@pytest.mark.parametrize("arch", ["sm_80", "sm_90a", "sm_100a"])
def test_smoke_accepts_actual_mma_lowering_without_requiring_newer_instructions(op, arch):
    result = instruction_evidence(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32;", dict(kind="cuda", arch=arch), op, dict(num_stages=2)
    )
    assert result["status"] == "verified"
    assert result["observed"]["mma.sync"]
    assert not result["observed"]["cp.async.bulk.tensor"]
    assert instruction_evidence("// mma.sync\nadd.f32;", dict(kind="cuda", arch=arch), op, {})["status"] == "missing"


@pytest.mark.parametrize("instruction", ["wgmma.mma_async", "tcgen05.mma"])
def test_smoke_records_alternative_native_matrix_instructions(instruction):
    result = instruction_evidence(instruction + ";", TARGETS["blackwell"], "gemm", {})
    assert result["status"] == "verified" and result["observed"][instruction]


def test_kda_intra_reductions_do_not_require_matrix_instructions():
    result = instruction_evidence("add.f32;", TARGETS["hopper"], "kda_chunk_intra_token_parallel", {})
    assert result["status"] == "not_applicable" and not result["required"]


def test_unsupported_fp8_carver_architecture_is_recorded_before_gpu_or_model_execution(tmp_path):
    workload = family_module("gemm_fp8", "cases").cases(holdout=True)[0]
    device = Device("ampere", TARGETS["ampere"])
    if FP8_COMPUTE_DTYPE == "bfloat16":
        assert carver_support_reason(workload, device) is None
        return
    assert "sm_89" in carver_support_reason(workload, device)
    result = run_native(make_request(workload, device, dict(method="carver")), tmp_path)
    assert result["status"] == "unsupported"
    assert "sm_89" in result["reason"]
    assert not list(tmp_path.iterdir())


def gpu(index, name, capability, utilization="0"):
    return dict(index=str(index), uuid=f"GPU-{index}", name=name, compute_cap=capability, **{"utilization.gpu": utilization})


def test_gpu_selection_routes_each_target_and_restores_visibility(monkeypatch):
    devices = [gpu(0, "NVIDIA H200", "9.0"), gpu(1, "NVIDIA A100", "8.0"), gpu(2, "NVIDIA B200", "10.0")]
    monkeypatch.setattr(monitor, "snapshot", lambda: dict(gpus=devices, processes=[]))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,1,0")
    for name, index in (("ampere", 1), ("blackwell", 2), ("hopper", 0)):
        with monitor.cuda_device(Device(name, TARGETS[name])):
            assert os.environ["CUDA_VISIBLE_DEVICES"] == f"GPU-{index}"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,1,0"
    with pytest.raises(RuntimeError, match="worker failed"), monitor.cuda_device(Device("ampere", TARGETS["ampere"])):
        raise RuntimeError("worker failed")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,1,0"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with monitor.cuda_device(Device("blackwell", TARGETS["blackwell"])):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-2"
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_gpu_selection_rejects_wrong_architecture_busy_and_hidden_devices(monkeypatch):
    devices = [gpu(0, "NVIDIA H200", "9.0"), gpu(1, "NVIDIA B200", "10.0", "100")]
    monkeypatch.setattr(monitor, "snapshot", lambda: dict(gpus=devices, processes=[]))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    for name in ("ampere", "blackwell"):
        with pytest.raises(RuntimeError, match="no matching idle"):
            monitor.select_cuda_gpu(Device(name, TARGETS[name]))
    devices[1]["utilization.gpu"] = "0"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="no matching idle"):
        monitor.select_cuda_gpu(Device("blackwell", TARGETS["blackwell"]))


def test_gpu_selection_allows_idle_coordinator_context_but_rejects_foreign_workers(monkeypatch):
    devices = [gpu(0, "NVIDIA A100", "8.0")]
    processes = [dict(pid=str(os.getpid()))]
    monkeypatch.setattr(monitor, "snapshot", lambda: dict(gpus=devices, processes=processes))
    monkeypatch.setattr(monitor, "foreign_processes", lambda *args: processes)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = Device("ampere", TARGETS["ampere"])
    assert monitor.select_cuda_gpu(device) == devices[0]
    processes.append(dict(pid=str(os.getpid() + 1)))
    with pytest.raises(RuntimeError, match="no matching idle"):
        monitor.select_cuda_gpu(device)


def test_baseline_collection_binds_each_target_before_observing_runtime(tmp_path, monkeypatch):
    from experiments.common import study

    devices = [gpu(0, "NVIDIA A100", "8.0"), gpu(1, "NVIDIA B200", "10.0")]
    monkeypatch.setattr(monitor, "snapshot", lambda: dict(gpus=devices, processes=[]))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    observed = []

    def runtime(device):
        observed.append((device.name, os.environ["CUDA_VISIBLE_DEVICES"]))
        return {}

    monkeypatch.setattr(study, "runtime_identity", runtime)
    # No cases: exercise routing without collecting any baselines.
    plan = dict(devices=[Device(name, TARGETS[name]).to_dict() for name in ("ampere", "blackwell")], unavailable={}, splits=dict(test=[]))
    assert study.execute(plan, tmp_path, {}, run_baselines=True) == 0
    assert observed == [("ampere", "GPU-0"), ("blackwell", "GPU-1")]
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
