import json
from types import SimpleNamespace

import pytest

from experiments.common import b200, run
from experiments.common.resource_policy import b200_post_compile_policy


def test_b300_selection_verifies_cuda_identity_and_preserves_nvml_name(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    gpu = dict(index="0", uuid="GPU-test", name="NVIDIA H800", compute_cap="10.3")
    identity = dict(uuid="GPU-test", name="NVIDIA B300 SXM6 AC", compute_cap="10.3")
    monkeypatch.setattr(b200.subprocess, "check_output", lambda *args, **kwargs: json.dumps([identity]))
    selected = b200.select_gpus(dict(gpus=[gpu]), [0], required=1, gpu_model="B300")
    assert selected == [dict(gpu, name=identity["name"], nvml_name=gpu["name"])]


@pytest.mark.parametrize(
    "name,capability,uuid",
    [("NVIDIA H800", "10.3", "GPU-test"), ("NVIDIA B300", "10.0", "GPU-test"), ("NVIDIA B300", "10.3", "GPU-other")],
)
def test_b300_selection_rejects_unverified_hardware(monkeypatch, name, capability, uuid):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    gpu = dict(index="0", uuid="GPU-test", name="NVIDIA H800", compute_cap="10.3")
    identity = dict(uuid=uuid, name=name, compute_cap=capability)
    monkeypatch.setattr(b200.subprocess, "check_output", lambda *args, **kwargs: json.dumps([identity]))
    with pytest.raises(ValueError):
        b200.select_gpus(dict(gpus=[gpu]), [0], required=1, gpu_model="B300")


@pytest.mark.parametrize("architecture", ["sm_103", "sm_103a"])
@pytest.mark.parametrize("operation", ["gemm", "attention", "kda_chunk_intra_token_parallel", "gemm_fp8", "grouped_gemm"])
def test_b300_uses_unchanged_b200_resource_limits(operation, architecture):
    workload = SimpleNamespace(op=operation)
    assert b200_post_compile_policy(workload, dict(kind="cuda", arch=architecture)) == b200_post_compile_policy(
        workload, dict(kind="cuda", arch="sm_100a")
    )


@pytest.mark.parametrize("metric", ["bound_aware", "rank_product", "work_max"])
def test_b300_cli_forwards_lightweight_metrics_without_profiles(tmp_path, monkeypatch, metric):
    requests = []

    def run_case(request, output):
        requests.append(request)
        return dict(status="analyzed")

    monkeypatch.setattr(run, "run_case", run_case)
    monkeypatch.setattr(run.sys, "argv", [
        "run", "--devices", "b300", "--workloads", "gemm_decode", "--metric", metric, "--output", str(tmp_path),
    ])
    for variable in ("TILELANG_DISABLE_CACHE", "TILELANG_AUTO_TUNING_DISABLE_CACHE", "TILELANG_AUTO_TUNING_CPU_COUNTS"):
        monkeypatch.setenv(variable, "1")
    assert run.main() == 0
    assert len(requests) == 1
    assert requests[0]["settings"]["metric"] == metric
    assert requests[0]["device"]["name"] == "b300"
    assert requests[0]["device"]["target"] == dict(kind="cuda", arch="sm_103a")


@pytest.mark.parametrize("architecture", ["sm_103", "sm_103a"])
def test_b300_carver_uses_the_existing_tensor_core_model(architecture):
    from experiments.gemm.carver import model_target

    assert str(model_target(dict(kind="cuda", arch=architecture)).attrs["arch"]) == "sm_90"
