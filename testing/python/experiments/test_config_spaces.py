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
    for preset in ("current", "expanded", "exhaustive"):
        space = configuration_space(replace(w, config_space=preset), d)
        configs = space["configs"]
        assert configs[: len(previous)] == previous
        assert len(configs) == len(set(space["config_ids"]))
        assert space["generated_count"] == len(configs) + space["rejected_count"] + space["alias_count"]
        assert space["compiled_count"] is None and space["correct_count"] is None
        assert all(a["index"] < len(configs) for a in space["aliases"])
        previous = configs
    assert len(previous) >= 200


@pytest.mark.parametrize("target", ["ampere", "hopper", "blackwell", "mi355x"])
@pytest.mark.parametrize("op", ["gemm", "attention", "kda_recurrent", "kda_chunk_o", "softmax", "elementwise"])
def test_large_caps_pools_preserving_current_configs_coverage_and_audit(target, op):
    w = next(w for w in default_workloads() if w.op == op)
    d = Device(target, TARGETS[target])
    current = configurations(w, d)
    full = configuration_space(replace(w, config_space="exhaustive"), d)
    compact = configuration_space(replace(w, config_space="large"), d)
    assert compact["configs"][: len(current)] == current
    assert compact["candidate_count"] == min(1024, full["candidate_count"])
    assert compact["generated_count"] == (
        compact["candidate_count"] + compact["budget_omitted_count"] + compact["alias_count"] + compact["rejected_count"]
    )
    if full["candidate_count"] <= 1024:
        assert compact["configs"] == full["configs"]
        assert "selection" not in compact
        return
    selection = compact["selection"]
    assert selection["covered_features"] == selection["total_features"]
    assert compact["configs"] == [full["configs"][i] for i in selection["indices"]]
    assert len(set(compact["config_ids"])) == len(compact["configs"])
    for alias in compact["aliases"]:
        if alias["index"] is None:
            assert alias["exhaustive_index"] not in selection["indices"]
        else:
            assert selection["indices"][alias["index"]] == alias["exhaustive_index"]


def test_large_retains_measured_gemm_and_attention_winners():
    d = Device("ampere", TARGETS["ampere"])
    gemm = replace(default_workloads()[0], config_space="large")
    pool = configurations(gemm, d)
    # Winners of the archived 4096^3 and 8192^3 complete A100 sweeps.
    for stages, panel in ((4, 0), (2, 8)):
        assert dict(block_m=128, block_n=128, block_k=32, stages=stages, threads=128, warp_policy="square", swizzle_panel=panel) in pool
    # Previously measured TileTune winners remain available too.
    for stages, panel in ((4, 0), (2, 8)):
        assert dict(block_m=128, block_n=256, block_k=16, stages=stages, threads=256, warp_policy="square", swizzle_panel=panel) in pool
    attention = replace(next(w for w in default_workloads() if w.op == "attention"), config_space="large")
    pool = configurations(attention, d)
    for m, stage, qk in ((32, 3, "square"), (64, 2, "full_row")):
        assert (
            dict(
                implementation="tiled", block_M=m, block_N=32, num_stages=stage, threads=256, qk_policy=qk, pv_policy="square", copy_width=8
            )
            in pool
        )
    assert pool == configurations(attention, d)


def test_large_retains_measured_kda_winners():
    w = replace(next(w for w in default_workloads() if w.op == "kda_chunk_o"), config_space="large")
    pool = configurations(w, Device("ampere", TARGETS["ampere"]))
    for m, v, intra, threads in ((64, 64, 3, 256), (16, 128, 2, 128)):
        assert (
            dict(implementation="tiled", block_m=m, block_k=16, block_v=v, block_s=16, stages=4, intra_stages=intra, threads=threads)
            in pool
        )


def test_large_preserves_oversized_explicit_pools():
    w = replace(default_workloads()[0], config_space="large", configs=[dict(native_tile=i) for i in range(1100)])
    d = Device("ampere", TARGETS["ampere"])
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
assert len(configurations(w, Device('ampere', TARGETS['ampere']))) == 1024
assert len(configurations(replace(w, config_space='exhaustive'), Device('ampere', TARGETS['ampere']))) == 6180
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
