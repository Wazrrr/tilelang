"""Expanded pools preserve identities, semantic workloads and partial labels."""

from dataclasses import replace
import json
import subprocess
import sys

import pytest

from experiments.portable.spec import Device, TARGETS, configuration_space, configurations, default_workloads
from experiments.portable.spaces import config_id
from experiments.portable.compare import training_sample
from experiments.xgboost.data import canonical_workload


def test_mma_policy_aliases_match_actual_compiler_partitioning():
    from itertools import product
    from tilelang.tileop.base import GemmWarpPolicy
    from experiments.portable.spaces import mma_partition

    for m, n, threads, policy in product((16, 32, 64, 128, 256), (16, 32, 64, 128, 256), (128, 256), GemmWarpPolicy):
        name = {GemmWarpPolicy.Square: "square", GemmWarpPolicy.FullRow: "full_row", GemmWarpPolicy.FullCol: "full_col"}[policy]
        assert mma_partition(m, n, threads, name) == policy.compute_warp_partition(m, n, threads // 32)


def test_compilation_census_does_not_count_failures_or_duplicate_programs_as_distinct():
    from experiments.portable.execution import compilation_census

    report = compilation_census(
        [
            dict(status="benchmarked", program_sha256="one"),
            dict(status="benchmarked", program_sha256="one"),
            dict(status="benchmark_error", program_sha256="two"),
            dict(status="compilation_failed"),
        ]
    )
    assert report["compiled_count"] == 3 and report["correct_count"] == 2
    assert report["distinct_program_count"] == 2 and report["distinct_correct_program_count"] == 1


@pytest.mark.parametrize("op", ["gemm", "attention", "kda_recurrent", "kda_chunk_o", "softmax", "rmsnorm", "reduce_sum", "elementwise"])
def test_nested_spaces_preserve_existing_indices_and_account_for_every_candidate(op):
    w = next(w for w in default_workloads() if w.op == op)
    d = Device("ampere", TARGETS["ampere"])
    previous = []
    for preset in ("current", "expanded", "large"):
        space = configuration_space(replace(w, config_space=preset), d)
        configs = space["configs"]
        assert configs[: len(previous)] == previous
        assert len(configs) == len(set(space["config_ids"]))
        assert space["generated_count"] == len(configs) + space["rejected_count"] + space["alias_count"]
        assert space["compiled_count"] is None and space["correct_count"] is None
        assert all(a["index"] < len(configs) for a in space["aliases"])
        previous = configs
    assert len(previous) >= 200


def test_presets_are_not_new_mathematical_workloads_and_do_not_expand_training_fraction():
    d = Device("ampere", TARGETS["ampere"])
    original = default_workloads()[0]
    w = replace(original, config_space="expanded")
    assert canonical_workload(original) == canonical_workload(w)
    sample = training_sample(w, d, fraction=0.1, seed=123)
    pool = configurations(w, d)
    assert len(sample["config_indices"]) == (len(pool) + 9) // 10
    assert len(sample["config_indices"]) < len(pool)
    assert sample["xgb_sampling"]["pool_configs"] == pool
    assert config_id(dict(a=1, b=2)) == config_id(dict(b=2, a=1))


def test_explicit_grids_are_not_silently_filtered():
    w = replace(default_workloads()[0], config_space="large", configs=[dict(threads=99999)])
    d = Device("ampere", TARGETS["ampere"])
    assert configurations(w, d) == w.configs
    assert configuration_space(w, d)["preset"] == "explicit"
    with pytest.raises(ValueError, match="unique configurations"):
        configurations(replace(w, configs=w.configs * 2), d)


@pytest.mark.parametrize("op", ["gemm", "attention"])
def test_invalid_explicit_configs_remain_available_for_recording_failures(op):
    w = next(w for w in default_workloads() if w.op == op)
    d = Device("ampere", TARGETS["ampere"])
    config = dict(configurations(w, d)[0], threads=0)
    explicit = replace(w, config_space="expanded", configs=[config])
    assert configurations(explicit, d) == [config]


def test_planning_large_spaces_does_not_import_compiler_or_ml_libraries():
    code = """
import sys
from dataclasses import replace
from experiments.portable.spec import *
w = replace(default_workloads()[0], config_space='large')
assert len(configurations(w, Device('ampere', TARGETS['ampere']))) > 1000
assert not {'torch','tilelang','xgboost','numpy'} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_comparison_plan_carries_preset_through_all_splits(capsys):
    from experiments.portable.compare import main

    assert main(["--plan", "--workloads", "gemm_nn", "--config-space", "expanded"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert all(w["config_space"] == "expanded" for split in plan["splits"].values() for w in split)
    assert plan["xgb_sampling"]["fraction"] == 0.1
    assert plan["settings"]["top_k"] == 20
