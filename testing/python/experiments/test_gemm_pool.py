"""One deterministic example-based GEMM pool is shared by every consumer."""

from dataclasses import replace

import pytest

from experiments.common.spec import Device, TARGETS, Workload, configuration_space
from experiments.gemm.spaces import get_configs
from experiments.suite import study_plan


@pytest.mark.parametrize("target", ["ampere", "hopper", "blackwell", "mi355x"])
def test_gemm_pool_is_complete_and_identical_across_targets(target):
    w = Workload("gemm", "gemm", dict(m=4096, n=4096, k=4096, transpose_b=True))
    space = configuration_space(w, Device(target, TARGETS[target]))
    assert space["preset"] == "expanded"
    assert space["configs"] == get_configs()
    assert space["candidate_count"] == len(set(space["config_ids"])) == 609
    assert space["rejected_count"] == space["alias_count"] == space["budget_omitted_count"] == 0
    assert "selection" not in space


@pytest.mark.parametrize("preset", ["current", "large", "exhaustive"])
def test_retired_gemm_presets_are_rejected(preset):
    with pytest.raises(ValueError, match="one configuration space"):
        w = Workload("gemm", "gemm", dict(m=128, n=128, k=128, transpose_b=True), config_space=preset)
        configuration_space(w, Device("hopper", TARGETS["hopper"]))


def test_explicit_gemm_configs_can_only_select_from_the_declared_pool():
    w = Workload("gemm", "gemm", dict(m=128, n=128, k=128, transpose_b=True))
    d = Device("hopper", TARGETS["hopper"])
    configs = get_configs()[13:16]
    assert configuration_space(replace(w, configs=configs), d)["configs"] == configs
    with pytest.raises(ValueError, match="subset"):
        configuration_space(replace(w, configs=[dict(configs[0], block_K=128)]), d)


@pytest.mark.parametrize(
    "parameters,dtype",
    [
        (dict(transpose_b=False), "float16"),
        (dict(transpose_a=True), "float16"),
        (dict(batch=2), "float16"),
        (dict(epilogue="bias_relu"), "float16"),
        ({}, "float32"),
    ],
)
def test_unsupported_gemm_never_selects_another_kernel(parameters, dtype):
    from experiments.gemm.kernel import make_case

    w = Workload("gemm", "gemm", dict(m=128, n=128, k=128, transpose_b=True) | parameters, dtype)
    with pytest.raises(ValueError, match="GEMM example"):
        make_case(w)


def test_final_study_uses_all_609_and_smoke_records_a_subset():
    device = Device("hopper", TARGETS["hopper"])
    final = study_plan("final", [device], families=["gemm"])
    smoke = study_plan("smoke", [device], families=["gemm"])
    assert all(len(c["indices"]) == 609 for c in final["subsets"]["hopper"].values())
    assert all(len(c["indices"]) == 16 for c in smoke["subsets"]["hopper"].values())
    assert all(w["config_space"] == "expanded" for split in final["splits"].values() for w in split)
