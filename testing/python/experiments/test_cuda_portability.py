"""One experiment definition across CUDA architectures, without GPU execution."""

import os

import pytest

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
        if op == "gemm_fp8" and name == "ampere":
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


@pytest.mark.parametrize("op", ["gemm", "attention"])
def test_carver_supports_cuda_architectures_and_rejects_other_backends_before_execution(op, tmp_path):
    workload = family_module(op, "cases").cases(holdout=True)[0]
    for name in ("ampere", "hopper", "blackwell"):
        assert carver_support_reason(workload, Device(name, TARGETS[name])) is None
    device = Device("mi355x", TARGETS["mi355x"])
    result = run_native(make_request(workload, device, dict(method="carver")), tmp_path)
    assert result["status"] == "unsupported"
    assert "CUDA" in result["reason"]
    assert not list(tmp_path.iterdir())


def gpu(index, name, capability, utilization="0"):
    return dict(index=str(index), uuid=f"GPU-{index}", name=name, compute_cap=capability, **{"utilization.gpu": utilization})


def test_gpu_selection_routes_each_target_without_mutating_visibility(monkeypatch):
    devices = [gpu(0, "NVIDIA H200", "9.0"), gpu(1, "NVIDIA A100", "8.0"), gpu(2, "NVIDIA B200", "10.0")]
    monkeypatch.setattr(monitor, "snapshot", lambda: dict(gpus=devices, processes=[]))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,1,0")
    for name, index in (("ampere", 1), ("blackwell", 2), ("hopper", 0)):
        assert monitor.select_cuda_gpu(Device(name, TARGETS[name]))["uuid"] == f"GPU-{index}"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,1,0"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert monitor.select_cuda_gpu(Device("blackwell", TARGETS["blackwell"]))["uuid"] == "GPU-2"
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


def test_baseline_probe_routes_each_target_without_rebinding_the_coordinator(monkeypatch):
    import json
    from experiments.utils import baseline_store

    devices = [gpu(0, "NVIDIA A100", "8.0"), gpu(1, "NVIDIA B200", "10.0")]
    monkeypatch.setattr(monitor, "snapshot", lambda: dict(gpus=devices, processes=[]))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    observed = []

    def probe(command, *, text, env):
        observed.append((json.loads(command[-1])["name"], env["CUDA_VISIBLE_DEVICES"]))
        return "{}"

    monkeypatch.setattr(baseline_store.subprocess, "check_output", probe)
    for name in ("ampere", "blackwell"):
        assert baseline_store.runtime_identity(Device(name, TARGETS[name])) == {}
    assert observed == [("ampere", "GPU-0"), ("blackwell", "GPU-1")]
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
