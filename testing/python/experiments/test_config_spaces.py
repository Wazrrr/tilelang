"""Expanded pools preserve identities, semantic workloads and partial labels."""

from dataclasses import replace
import json
import subprocess
import sys

import pytest

from experiments.common.spec import Device, TARGETS, configuration_space, configurations, default_workloads
from experiments.common.spaces import config_id
from experiments.common.comparison import training_sample
from experiments.xgboost.data import canonical_workload


def test_compilation_census_does_not_count_failures_or_duplicate_programs_as_distinct():
    from experiments.common.execution import compilation_census

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


def test_expanded_retains_attention_example_launches():
    d = Device("ampere", TARGETS["ampere"])
    attention = next(w for w in default_workloads() if w.op == "attention")
    pool = configurations(attention, d)
    for size in (64, 128):
        assert dict(block_M=size, block_N=size, num_stages=1, threads=128) in pool
    assert pool == configurations(attention, d)


def test_expanded_retains_native_kda_example_grid():
    from examples.kda.chunk_intra_token_parallel import get_configs

    w = next(w for w in default_workloads() if w.op == "kda_chunk_intra_token_parallel")
    pool = configurations(w, Device("ampere", TARGETS["ampere"]))
    for c in get_configs():
        assert c in pool


def test_explicit_subsets_are_not_capped():
    d = Device("ampere", TARGETS["ampere"])
    original = default_workloads()[0]
    w = replace(original, configs=configurations(original, d)[:1100])
    assert configurations(w, d) == w.configs
    assert configurations(replace(w, configs=None), replace(d, configs={w.name: w.configs})) == w.configs


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
    w = replace(default_workloads()[0], configs=[dict(native_tile=99999)])
    d = Device("ascend910b", TARGETS["ascend910b"])
    assert configurations(w, d) == w.configs
    assert configuration_space(w, d)["preset"] == "explicit"
    with pytest.raises(ValueError, match="unique configurations"):
        configurations(replace(w, configs=w.configs * 2), d)


@pytest.mark.parametrize("op", ["attention"])
def test_explicit_configs_cannot_extend_the_fixed_example_pool(op):
    w = next(w for w in default_workloads() if w.op == op)
    d = Device("ampere", TARGETS["ampere"])
    config = dict(configurations(w, d)[0], threads=0)
    explicit = replace(w, config_space="expanded", configs=[config])
    with pytest.raises(ValueError, match="subset"):
        configurations(explicit, d)


def test_planning_large_spaces_does_not_import_compiler_or_ml_libraries():
    code = """
import sys
from dataclasses import replace
from experiments.common.spec import *
w = default_workloads()[0]
assert w.config_space == 'expanded'
assert len(configurations(w, Device('ampere', TARGETS['ampere']))) == 576
assert not {'torch','tilelang','xgboost','numpy'} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_comparison_plan_carries_preset_through_all_splits(capsys):
    from experiments.common.comparison import main

    assert main(["--plan", "--workloads", "gemm_square", "--config-space", "expanded"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert all(w["config_space"] == "expanded" for split in plan["splits"].values() for w in split)
    assert plan["xgb_sampling"]["fraction"] == 0.1
    assert plan["settings"]["top_k"] == 20
