"""Each non-GEMM final family has one complete pool of example parameters."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from experiments.common.spec import Device, TARGETS, Workload, configuration_space, load_manifest
from experiments.families import family_module
from experiments.suite import core_cases, study_plan

COUNTS = {"attention": 192, "kda_chunk_o": 720, "gemm_fp8": 2304, "grouped_gemm": 2}
CASES = [w for w in core_cases("final") if w.op in COUNTS]
REPRESENTATIVES = [next(w for w in CASES if w.op == op) for op in COUNTS]


@pytest.mark.parametrize("target", ["ampere", "hopper", "blackwell", "mi355x"])
@pytest.mark.parametrize("w", CASES, ids=lambda w: w.name)
def test_complete_pool_has_stable_ids_without_cap_or_prefilter(w, target):
    space = configuration_space(w, Device(target, TARGETS[target]))
    assert space["preset"] == "expanded"
    assert space["configs"] == family_module(w.op, "spaces").get_configs()
    assert len(set(space["config_ids"])) == space["candidate_count"] == COUNTS[w.op]
    assert space["retained_current_count"] == space["rejected_count"] == space["alias_count"] == space["budget_omitted_count"] == 0
    assert "selection" not in space


@pytest.mark.parametrize("preset", ["current", "large", "exhaustive"])
@pytest.mark.parametrize("w", REPRESENTATIVES, ids=lambda w: w.op)
def test_retired_presets_are_rejected(w, preset):
    with pytest.raises(ValueError, match="one configuration space"):
        configuration_space(replace(w, config_space=preset), Device("hopper", TARGETS["hopper"]))


@pytest.mark.parametrize("w", REPRESENTATIVES, ids=lambda w: w.op)
def test_explicit_native_configs_must_select_from_same_pool(w):
    device = Device("hopper", TARGETS["hopper"])
    pool = family_module(w.op, "spaces").get_configs()
    selected = pool if len(pool) < 10 else pool[7:10]
    assert configuration_space(replace(w, configs=selected), device)["configs"] == selected
    with pytest.raises(ValueError, match="subset"):
        configuration_space(replace(w, configs=[dict(pool[0], threads=123)]), device)
    with pytest.raises(ValueError, match="subset"):
        configuration_space(w, replace(device, configs={w.name: [dict(pool[0], implementation="legacy")]}))


def test_every_example_config_and_explicit_default_is_included():
    from examples.kda.chunk_o import get_configs as kda_configs

    fa = family_module("attention", "spaces").get_configs()
    kda = family_module("kda_chunk_o", "spaces").get_configs()
    fp8 = family_module("gemm_fp8", "spaces").get_configs()
    assert dict(block_M=128, block_N=128, num_stages=1, threads=128) in fa
    assert len(kda_configs()) == 90 and len(kda) == 8 * len(kda_configs())
    assert all(c in kda for c in kda_configs())
    assert dict(block_DK=64, block_DV=64, num_stages=0, threads=256) in kda
    assert len(fp8) == 2304
    assert dict(block_M=32, block_N=192, block_K=128, num_stages=2, threads=256, enable_rasteration=True) in fp8
    assert dict(block_M=64, block_N=256, block_K=32, num_stages=2, threads=256, enable_rasteration=False) in fp8
    grouped = family_module("grouped_gemm", "spaces").get_configs()
    assert grouped == [
        dict(block_M=128, block_N=256, block_K=128, num_stages=6, threads=128, persistent=False),
        dict(block_M=128, block_N=256, block_K=128, num_stages=6, threads=256, persistent=True),
    ]


def test_every_case_split_and_frozen_manifest_use_the_single_pool():
    root = Path(__file__).resolve().parents[3]
    _, frozen = load_manifest(json.loads((root / "experiments/manifests/five_target_final.json").read_text()))
    assert frozen == core_cases("final")
    plan = study_plan("final", [Device("hopper", TARGETS["hopper"])])
    for split in plan["splits"].values():
        assert all(w["config_space"] == "expanded" for w in split)
    for w in CASES:
        assert len(plan["subsets"]["hopper"][w.name]["indices"]) == COUNTS[w.op]
    smoke = study_plan("smoke", [Device("hopper", TARGETS["hopper"])])
    assert all(
        len(smoke["subsets"]["hopper"][w.name]["indices"]) == min(16, COUNTS[w.op])
        for w in CASES
        if w.name in smoke["subsets"]["hopper"]
    )


def test_recurrent_kda_is_not_silently_substituted_by_chunk_output():
    with pytest.raises(ValueError, match="Unknown operation"):
        Workload("retired", "kda_recurrent", dict(batch=1, heads=1, sequence=64, dim=64, value_dim=64))
