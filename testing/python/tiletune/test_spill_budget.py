"""Physical occupancy is strict; conservative tile demand has a separate margin."""

import pytest

from tilelang.tiletune import TileTuneConfig, analyze_prim_func, check_compiler_resources
from tilelang.tiletune.register_policy import analyze_register_policy
from tilelang.tiletune.families import AttentionSpecialization, GemmSpecialization
from tilelang.autotuner.filters.launch import LaunchResourceInfo
from tilelang.contrib.cuda_resource_info import parse_ptxas_output
from examples.flash_attention.example_mha_tiletune import TARGET, make_attention
from test_analysis import gemm
from test_cost import LIMITS
from test_modules import PROFILE


POLICY = dict(
    status="predicted",
    producer_threads=128,
    consumer_threads=256,
    launch_threads=384,
    producer_register_request=24,
    consumer_register_request=240,
    register_reservation_per_block=64512,
)


def policy_result(budget=32, *, policy=None, tile=65920, lower=128, limits=None, family="attention", matched=True):
    return analyze_register_policy(
        dict(
            warp_specialization=POLICY if policy is None else policy,
            budget=255,
            budget_source="architecture register limit",
            hardware_register_cap=255,
            modeled_lower_bound=lower,
            tile_liveness=dict(peak_registers_per_block_estimate=tile, computing_threads_estimate=256),
        ),
        TileTuneConfig(attention_spill_budget_registers_per_thread=budget),
        (AttentionSpecialization if family == "attention" else GemmSpecialization)(name=family, matched=matched),
        LIMITS if limits is None else limits,
    )


@pytest.mark.parametrize(
    "budget,status", [(0, "exceeds_allowance"), (17, "exceeds_allowance"), (18, "within_allowance"), (32, "within_allowance")]
)
def test_budget_respects_consumer_partition_without_changing_occupancy(budget, status):
    result = policy_result(budget)
    physical, demand = result["physical_register_allocation"], result["register_demand"]
    assert physical["registers_per_block"] == 64512
    assert physical["resident_blocks_register_limit"] == 1
    assert physical["status"] == "within_limit"
    assert demand["status"] == status
    assert demand["registers_per_thread_estimate"] == 258
    assert demand["capacity_registers_per_thread"] == 240
    assert demand["excess_registers_per_thread_estimate"] == 18
    assert demand["excess_bytes_per_thread_estimate"] == 72
    # Conservative liveness can gate the score, never reject on its own.
    assert result["decision"]["keep"]


def test_hard_capacity_cannot_be_enlarged_by_allowance():
    result = policy_result(1000, limits={**LIMITS, "registers_per_sm": 64000})
    assert result["physical_register_allocation"]["resident_blocks_register_limit"] == 0
    assert not result["decision"]["keep"]
    policy = dict(POLICY, consumer_register_request=256, register_reservation_per_block=68608)
    result = policy_result(1000, policy=policy, limits={**LIMITS, "registers_per_sm": 100000})
    assert result["physical_register_allocation"]["status"] == "exceeds_limit"
    assert not result["decision"]["keep"]  # per-thread hardware ceiling also strict


@pytest.mark.parametrize(
    "policy", [dict(POLICY, status="unknown"), dict(POLICY, consumer_threads=None), dict(POLICY, register_reservation_per_block=1)]
)
def test_unresolved_policy_is_not_a_physical_allocation(policy):
    result = policy_result(policy=policy)
    assert result["physical_register_allocation"]["status"] == "unknown"
    assert result["physical_register_allocation"]["registers_per_block"] is None
    assert result["decision"]["keep"]


def test_unknown_inputs_remain_unknown():
    assert policy_result(tile=None)["register_demand"]["status"] == "unknown"
    assert policy_result(limits={})["physical_register_allocation"]["status"] == "unknown"


@pytest.mark.parametrize("causal", [False, True])
def test_attention_winner_physical_occupancy_is_independent_of_soft_scoring(causal):
    func = make_attention(causal=causal)(block_M=128, block_N=256, num_stages=1, threads=256)
    settings = dict(ranking_metric="pipeline_time", performance_model=PROFILE, max_spill_bytes=None, max_local_bytes=None)
    before = analyze_prim_func(func, settings, target=TARGET, device_limits=LIMITS)
    after = analyze_prim_func(func, dict(settings, attention_spill_budget_registers_per_thread=32), target=TARGET, device_limits=LIMITS)
    assert before["tile_cost"]["score"] is None
    assert after["tile_cost"]["score"] > 0
    assert before["modules"]["waves"] == after["modules"]["waves"]
    assert before["pressure"]["physical_register_allocation"] == after["pressure"]["physical_register_allocation"]
    cost = after["tile_cost"]
    assert cost["logical_tile_registers_per_block_estimate"] == 65920
    assert cost["registers_per_block_estimate"] == 64512
    assert cost["resident_blocks_per_sm_estimate"] == 1
    assert cost["num_waves_estimate"] == 4
    assert after["modules"]["ranking"]["conditional_on_spill_allowance"]
    assert after["modules"]["ranking"]["spill_traffic_modeled"] is False
    assert after["pressure"]["decision"] == before["pressure"]["decision"]
    assert after["pressure"]["modeled_lower_bound"] == before["pressure"]["modeled_lower_bound"] == 128
    assert after["modules"]["memory_traffic"] == before["modules"]["memory_traffic"]


@pytest.mark.parametrize("stages", [0, 1])
@pytest.mark.parametrize("ranking", [False, True])
def test_proven_demand_uses_soft_limit_even_without_ranking(stages, ranking):
    factory = make_attention()
    settings = dict(attention_spill_budget_registers_per_thread=32, ranking=ranking)
    func = factory(block_M=128, block_N=256, num_stages=stages, threads=128)
    strict = analyze_prim_func(func, dict(settings, attention_spill_budget_registers_per_thread=0), target=TARGET, device_limits=LIMITS)
    relaxed = analyze_prim_func(func, settings, target=TARGET, device_limits=LIMITS)
    assert strict["pressure"]["modeled_lower_bound"] == relaxed["pressure"]["modeled_lower_bound"] == 256
    assert not strict["pressure"]["decision"]["keep"]
    assert relaxed["pressure"]["decision"]["keep"]
    assert relaxed["pressure"]["register_demand"]["status"] == "exceeds_allowance"  # estimate still too large to score
    oversized = factory(block_M=256, block_N=256, num_stages=stages, threads=128)
    assert not analyze_prim_func(oversized, settings, target=TARGET, device_limits=LIMITS)["pressure"]["decision"]["keep"]
    report = analyze_prim_func(oversized, dict(settings, mode="report_only"), target=TARGET, device_limits=LIMITS)
    assert report["pressure"]["decision"]["keep"] and report["pressure"]["decision"]["would_reject"]


@pytest.mark.parametrize("limits", [{**LIMITS, "max_threads_per_sm": 256}, {**LIMITS, "shared_memory_per_sm": 65536}])
def test_soft_allowance_does_not_bypass_other_occupancy_limits(limits):
    func = make_attention()(block_M=128, block_N=256, num_stages=1, threads=256)
    result = analyze_prim_func(func, {"attention_spill_budget_registers_per_thread": 32}, target=TARGET, device_limits=limits)
    assert result["pressure"]["register_demand"]["status"] == "within_allowance"
    assert result["tile_cost"]["resident_blocks_per_sm_estimate"] == 0
    assert result["tile_cost"]["score"] is None


def test_allowance_does_not_supply_missing_profile():
    func = make_attention()(block_M=128, block_N=256, num_stages=1, threads=256)
    result = analyze_prim_func(
        func, {"attention_spill_budget_registers_per_thread": 32, "ranking_metric": "pipeline_time"}, target=TARGET, device_limits=LIMITS
    )
    assert result["tile_cost"]["resident_blocks_per_sm_estimate"] == 1
    assert result["tile_cost"]["score"] is None


def test_tuner_resolves_physical_limits_without_ranking(monkeypatch):
    from tilelang.autotuner import AutoTuner

    def queried(target):
        raise RuntimeError("physical limits requested")

    monkeypatch.setattr("tilelang.tiletune.query_device_limits", queried)
    tuner = AutoTuner(gemm, [{}]).set_compile_args(target=TARGET, execution_backend="tvm_ffi").set_tiletune_args(True, ranking=False)
    with pytest.raises(RuntimeError, match="physical limits requested"):
        tuner.run(early_stop=False)


def test_gemm_margin_is_zero_and_attention_setting_has_cache_identity():
    func = gemm(stages=1)
    base = analyze_prim_func(func, target=TARGET, device_limits=LIMITS)
    result = analyze_prim_func(func, {"attention_spill_budget_registers_per_thread": 32}, target=TARGET, device_limits=LIMITS)
    assert result == base
    assert not policy_result(1000, family="gemm", lower=256)["decision"]["keep"]
    assert policy_result(1000, family="gemm")["register_demand"]["allowance_registers_per_thread"] == 0
    assert TileTuneConfig(attention_spill_budget_registers_per_thread=32).to_cache_key_dict() != TileTuneConfig().to_cache_key_dict()


def test_unmatched_attention_cannot_use_the_family_spill_allowance():
    matched = policy_result(32, lower=256)
    unmatched = policy_result(32, lower=256, matched=False)
    assert matched["decision"]["keep"]
    assert not unmatched["decision"]["keep"]
    assert unmatched["register_demand"]["allowance_registers_per_thread"] == 0
    assert unmatched["physical_register_allocation"] == matched["physical_register_allocation"]


@pytest.mark.parametrize("registers,reject", [(168, False), (172, True)])
def test_compiler_initial_allocation_uses_strict_sm_capacity(registers, reject):
    resources = parse_ptxas_output(f"""ptxas info : Compiling entry function 'kernel' for 'sm_90'
ptxas info : Function properties for kernel
    48 bytes stack frame, 48 bytes spill stores, 48 bytes spill loads
ptxas info : Used {registers} registers
""")
    settings = dict(attention_spill_budget_registers_per_thread=1000, max_spill_bytes=None, max_local_bytes=None)
    launches = [LaunchResourceInfo(function_name="kernel", block_dims=(384, 1, 1))]
    result = check_compiler_resources(resources, ["kernel"], settings, target=TARGET, launch_infos=launches, device_limits=LIMITS)
    assert result["would_reject"] == reject
    assert result["physical_register_allocation"]["kernel"]["initial_registers_per_block_lower_bound"] == registers * 384
    missing = check_compiler_resources(resources, ["kernel"], settings, target=TARGET, launch_infos=[], device_limits=LIMITS)
    assert missing["status"] == "unknown" and missing["keep"]
    assert missing["physical_register_allocation"]["kernel"]["initial_registers_per_block_lower_bound"] is None
    report = check_compiler_resources(
        resources, ["kernel"], dict(settings, mode="report_only"), target=TARGET, launch_infos=launches, device_limits=LIMITS
    )
    assert report["keep"] and report["would_reject"] == reject


@pytest.mark.parametrize("budget", [-1, None, 1.5, True])
def test_invalid_attention_spill_budget(budget):
    with pytest.raises(ValueError, match="attention_spill_budget_registers_per_thread"):
        TileTuneConfig(attention_spill_budget_registers_per_thread=budget)
