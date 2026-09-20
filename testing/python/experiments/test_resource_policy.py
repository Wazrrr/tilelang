"""Explicit H200 experiment budgets never become kernel-family scoring hints."""

from types import SimpleNamespace
import json

import pytest

from experiments.common.resource_policy import h200_post_compile_policy
from tilelang.contrib.cuda_resource_info import parse_ptxas_output
from tilelang.tiletune import TileTuneConfig, check_compiler_resources


@pytest.mark.parametrize(
    "op,spill,local", [("gemm", 0, 0), ("grouped_gemm", 0, 0), ("kda_chunk_o", 0, 0), ("attention", 48, 48), ("gemm_fp8", 92, 64)]
)
def test_observed_h200_oracle_resources_fit_declared_policy(op, spill, local):
    target = dict(kind="cuda", arch="sm_90a")
    policy = h200_post_compile_policy(SimpleNamespace(op=op), target)
    config = TileTuneConfig(mode="report_only", max_spill_bytes=None, max_local_bytes=None, post_compile_policy=policy)
    report = parse_ptxas_output(f"""ptxas info : Compiling entry function 'kernel' for 'sm_90'
ptxas info : Function properties for kernel
    {local} bytes stack frame, {spill} bytes spill stores, {spill} bytes spill loads
ptxas info : Used 168 registers
""")
    assert check_compiler_resources(report, ["kernel"], config, target=target)["keep"]
    report["kernel"].extra["spill_loads_bytes"] = policy["max_spill_bytes"] + 1
    assert not check_compiler_resources(report, ["kernel"], config, target=target)["keep"]


@pytest.mark.parametrize("target", [dict(kind="cuda", arch="sm_80"), dict(kind="cuda", arch="sm_100a"), dict(kind="hip", mcpu="gfx950")])
def test_h200_limits_are_not_applied_to_other_backends(target):
    assert h200_post_compile_policy(SimpleNamespace(op="attention"), target) is None


def test_h200_memory_runner_records_separate_compiler_policy(tmp_path):
    from experiments.common.run import make_request, run_native
    from experiments.common.spec import Device, Workload

    workload = Workload("policy_gemm", "gemm", dict(m=64, n=64, k=64, transpose_b=True))
    device = Device("hopper", dict(kind="cuda", arch="sm_90a"), device_limits={"sm_count": 132})
    settings = dict(method="analyze", metric="memory", memory_regime="streaming", top_k=2, config_indices=[0], trace=False, seed=123)
    result = run_native(make_request(workload, device, settings), tmp_path)
    assert result["status"] == "analyzed"
    report = json.loads((tmp_path / "tiletune.json").read_text())
    assert report["settings"]["post_compile_policy"] == dict(mode="reject", max_spill_bytes=0, max_local_bytes=0)
    assert report["settings"]["mode"] == "report_only"
    assert report["settings"]["max_spill_bytes"] is None
    assert report["configs"][0]["tile_cost"]["score"] is not None
