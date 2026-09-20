"""Compiler-only resource policy must not change memory ranking or eligibility."""

from dataclasses import replace

import pytest

from tilelang.autotuner.filters.launch import LaunchResourceInfo
from tilelang.contrib.cuda_resource_info import parse_ptxas_output
from tilelang.tiletune import TileTuneConfig, TileTuneReject, analyze_prim_func, check_compiler_resources
from tilelang.tiletune.runtime import TileTuneSession
from test_analysis import gemm
from test_cost import LIMITS
from test_modules import TARGET


def resources(*, stores=0, loads=0, local=0, registers=168):
    return parse_ptxas_output(f"""ptxas info : Compiling entry function 'kernel' for 'sm_90'
ptxas info : Function properties for kernel
    {local} bytes stack frame, {stores} bytes spill stores, {loads} bytes spill loads
ptxas info : Used {registers} registers
""")


def config():
    return TileTuneConfig(
        enabled=True,
        ranking_metric="memory",
        mode="report_only",
        max_spill_bytes=None,
        max_local_bytes=None,
        post_compile_policy=dict(mode="reject", max_spill_bytes=64, max_local_bytes=64),
    )


def test_compiler_policy_preserves_pre_lowering_analysis():
    original = replace(config(), post_compile_policy=None)
    changed = replace(original, post_compile_policy=dict(mode="reject", register_cap=1, max_spill_bytes=0, max_local_bytes=0))
    func = gemm()
    before, after = [analyze_prim_func(func, settings, target=TARGET, device_limits=LIMITS) for settings in (original, changed)]
    assert before["tile_cost"] == after["tile_cost"]
    assert before["pressure"]["decision"] == after["pressure"]["decision"]
    assert before["tile_propagation"] == after["tile_propagation"]
    assert changed.mode == "report_only" and changed.max_spill_bytes is None
    assert original.to_cache_key_dict() != changed.to_cache_key_dict()
    assert not check_compiler_resources(resources(), ["kernel"], changed, target=TARGET)["keep"]


@pytest.mark.parametrize("field", ["stores", "loads", "local"])
def test_compiler_limits_are_inclusive_and_enforced_in_outer_report_mode(field):
    settings = config()
    allowed = check_compiler_resources(resources(**{field: 64}), ["kernel"], settings, target=TARGET)
    rejected = check_compiler_resources(resources(**{field: 65}), ["kernel"], settings, target=TARGET)
    assert allowed["keep"] and not allowed["would_reject"]
    assert not rejected["keep"] and rejected["would_reject"]
    assert rejected["classification"] == "policy_rejection"
    assert rejected["policy"] == dict(mode="reject", register_cap=None, max_spill_bytes=64, max_local_bytes=64)


@pytest.mark.parametrize("registers,threads,shared", [(256, 128, 0), (168, 512, 0), (32, 2048, 0), (32, 128, 240000)])
def test_spill_allowance_does_not_relax_physical_limits(registers, threads, shared):
    launch = LaunchResourceInfo(function_name="kernel", block_dims=(threads, 1, 1), dynamic_smem_bytes=shared)
    decision = check_compiler_resources(
        resources(stores=48, loads=48, local=48, registers=registers),
        ["kernel"],
        config(),
        target=TARGET,
        launch_infos=[launch],
        device_limits=LIMITS,
    )
    assert not decision["keep"]
    assert decision["physical_reasons"]
    assert decision["classification"] == "resource_violation"


def test_session_records_post_compile_rejection_without_replacement():
    session = TileTuneSession(config(), [{"block": 32}, {"block": 64}], target=TARGET)
    session.selection = dict(selected_indices=[0], selected_count=1)
    with pytest.raises(TileTuneReject, match="spill_loads_bytes"):
        session.post_compile(0, resources(loads=65), None)
    assert session.records[0]["status"] == "post_compile_rejected"
    assert session.records[0]["post_compile"]["policy"]["mode"] == "reject"
    assert session.records[1]["status"] == "pending"
    assert session.selection == dict(selected_indices=[0], selected_count=1)


def test_missing_compiler_counters_remain_unknown():
    decision = check_compiler_resources({}, ["kernel"], config(), target=TARGET)
    assert decision["status"] == "unknown"
    assert decision["resources"]["kernel"]["spill_loads_bytes"] is None


@pytest.mark.parametrize(
    "policy",
    [False, [], {"alpha": 0.5}, {"mode": "rank"}, {"max_spill_bytes": -1}, {"max_local_bytes": True}, {"register_cap": 0}],
)
def test_invalid_post_compile_policy_fails_at_configuration(policy):
    with pytest.raises(ValueError):
        TileTuneConfig(post_compile_policy=policy)
