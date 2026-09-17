"""Suite membership, original identities, and acceptance without measurements."""

from pathlib import Path
import subprocess
import sys

import pytest

from experiments.common.acceptance import assess_seed
from experiments.common.spec import Device, TARGETS, Workload
from experiments.utils.subsets import pairwise_subset
from experiments.suite import BUDGETS, CORE_TARGETS, core_cases, study_plan


def test_smoke_uses_same_cases_and_three_fixed_budgets():
    assert len(core_cases("smoke")) == 4
    assert {w.op for w in core_cases("smoke")} == {"gemm", "attention", "kda_chunk_o", "gemm_fp8"}
    assert all(w in core_cases("development") for w in core_cases("smoke"))
    assert len(core_cases("final")) == len(core_cases("development")) == 20
    assert all(sum(w.op == op for w in core_cases("final")) == 5 for op in {w.op for w in core_cases("final")})
    assert {w.dtype for w in core_cases("final")} == {"float16", "float8_e4m3fn", "float8_e5m2"}
    for w in core_cases("final"):
        if w.op == "kda_chunk_o":
            assert w.parameters["sequence"] % w.parameters["chunk_size"] == 0
    assert BUDGETS["final"]["seeds"] == [123, 456, 789]
    assert core_cases("full") == core_cases("final")
    assert BUDGETS["full"]["configurations"] is None
    assert "mi355x" in CORE_TARGETS and "mi308" not in CORE_TARGETS
    assert TARGETS["mi355x"]["mcpu"] == "gfx950"


def test_final_cases_use_five_common_serving_shapes_per_family():
    by_op = {
        op: [w for w in core_cases("final") if w.op == op]
        for op in ("gemm", "gemm_fp8", "attention", "kda_chunk_o")
    }
    dense_shapes = [
        (128, 4096, 4096),
        (1024, 4096, 4096),
        (1024, 4096, 14336),
        (4096, 4096, 4096),
        (4096, 14336, 4096),
    ]
    for op in ("gemm", "gemm_fp8"):
        assert [(w.parameters["m"], w.parameters["n"], w.parameters["k"]) for w in by_op[op]] == dense_shapes
    assert [(w.parameters["sequence"], w.parameters["causal"]) for w in by_op["attention"]] == [
        (512, True),
        (2048, True),
        (4096, False),
        (4096, True),
        (8192, True),
    ]
    assert all((w.parameters["heads"], w.parameters["dim"]) == (32, 128) for w in by_op["attention"])
    assert [(w.parameters["batch"], w.parameters["sequence"]) for w in by_op["kda_chunk_o"]] == [
        (1, 2048),
        (1, 4096),
        (1, 8192),
        (2, 4096),
        (1, 16384),
    ]
    assert all(
        (w.parameters["heads"], w.parameters["dim"], w.parameters["value_dim"], w.parameters["chunk_size"])
        == (64, 128, 128, 64)
        for w in by_op["kda_chunk_o"]
    )


def test_pairwise_is_deterministic_and_preserves_indices():
    w = Workload("fp8", "gemm_fp8", dict(m=64, n=96, k=128, transpose_b=True), dtype="float8_e4m3fn")
    d = Device("ampere", TARGETS["ampere"])
    configs = [dict(BLOCK_M=r, BLOCK_N=8192, threads=t) for r in [1, 2, 4] for t in [64, 128, 256]]
    first = pairwise_subset(w, d, configs, 5)
    reverse = pairwise_subset(w, d, configs[::-1], 5)
    assert set(first["config_ids"]) == set(reverse["config_ids"])
    assert first["indices"] == sorted(first["indices"])
    assert first == pairwise_subset(w, d, configs, 5)
    assert first["actual_pool_size"] == 5


def test_duplicate_configurations_never_fill_a_budget():
    w = Workload("fp8", "gemm_fp8", dict(m=128, n=128, k=128, transpose_b=True), dtype="float8_e4m3fn")
    d = Device("ampere", TARGETS["ampere"])
    c = dict(BLOCK_M=1, BLOCK_N=8192, threads=128)
    subset = pairwise_subset(w, d, [c, dict(c)], 16)
    assert subset["indices"] == [0]
    assert subset["actual_pool_size"] == 1
    assert subset["aliases"] == [dict(index=1, representative=0)]


def test_required_subset_members_use_the_budget_and_seed_coverage():
    w = Workload("fp8", "gemm_fp8", dict(m=64, n=96, k=128, transpose_b=True), dtype="float8_e4m3fn")
    d = Device("ampere", TARGETS["ampere"])
    configs = [dict(BLOCK_M=r, BLOCK_N=8192, threads=t) for r in (1, 2, 4) for t in (64, 128, 256)]
    subset = pairwise_subset(w, d, configs, 5, required_indices=[0, 8])
    assert {0, 8} <= set(subset["indices"])
    assert subset["actual_pool_size"] == 5
    reverse = pairwise_subset(w, d, configs[::-1], 5, required_indices=[0, 8])
    assert set(subset["config_ids"]) == set(reverse["config_ids"])
    with pytest.raises(ValueError, match="exceed"):
        pairwise_subset(w, d, configs, 1, required_indices=[0, 8])
    with pytest.raises(ValueError, match="inside"):
        pairwise_subset(w, d, configs, 5, required_indices=[9])


def test_no_cube_grid_is_invented_and_no_device_is_removed():
    d = Device("ascend910b", TARGETS["ascend910b"])
    plan = study_plan("smoke", [d])
    assert len(plan["devices"]) == 1
    assert len(plan["splits"]["test"]) == 4
    assert "ascend910b" in plan["unavailable"]
    assert plan["subsets"]["ascend910b"] == {}


def test_missing_case_fails_even_if_other_seven_are_good():
    cases = [w.to_dict() for w in core_cases("final")]
    rows = [
        dict(
            workload=w,
            methods=dict(tiletune=dict(status="completed", correctness="passed", tuning_seconds=1), brute_force=dict(tuning_seconds=2)),
            diagnostics=dict(tiletune=dict(correct_score_coverage=0.95, curves={"20": dict(oracle_at_k=0.98)})),
            validation=dict(tiletune=dict(samples_ms=[1] * 7)),
        )
        for w in cases
    ]
    assert assess_seed(cases, rows)["accepted"]
    assert not assess_seed(cases, rows[:-1])["accepted"]
    rows[0]["diagnostics"]["tiletune"]["correct_score_coverage"] = 0.89
    assert not assess_seed(cases, rows)["accepted"]


def test_planning_has_no_compiler_runtime_imports():
    root = str(Path(__file__).resolve().parents[3])
    code = (
        f"import sys; sys.path.insert(0, {root!r}); from experiments.suite import core_cases; "
        "assert len(core_cases('final')) == 20; "
        "assert not any(k.split('.')[0] in ('tilelang', 'tvm', 'torch') for k in sys.modules)"
    )
    subprocess.run([sys.executable, "-I", "-S", "-c", code], check=True)


def test_comparison_retains_per_split_config_overrides(tmp_path, capsys):
    from dataclasses import replace
    import json
    from experiments.common.comparison import main

    cases = [replace(core_cases("smoke")[0], name=name).to_dict() for name in ("training", "validation", "test")]
    for i, item in enumerate(cases):
        item["parameters"] = dict(m=128 + i * 128, n=128, k=128)
    configs = [dict(block_M=32, block_N=32, block_K=32, num_stages=0, thread_num=128, enable_rasteration=False)]
    device = Device("ampere", TARGETS["ampere"], configs={w["name"]: configs for w in cases}, subsets={w["name"]: [0] for w in cases})
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps(dict(version=1, devices=[device.to_dict()], splits=dict(zip(("train", "validation", "test"), [[w] for w in cases]))))
    )
    assert main(["--split-manifest", str(path), "--plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["devices"][0]["configs"] == device.configs
    assert plan["devices"][0]["subsets"] == device.subsets


def test_protocol_versions_preserve_archived_wire_identity():
    import hashlib
    import json
    from experiments.common.run import make_request, validate_request

    w = core_cases("smoke")[0]
    d = Device("ampere", TARGETS["ampere"])
    request = make_request(w, d, dict(method="analyze"))
    assert request["version"] == 1
    assert validate_request(request) == (w, d)
    request = make_request(w, d, dict(method="smoke"))
    assert request["version"] == 2
    assert validate_request(request) == (w, d)
    # Early archived requests may use v1 with the same optional fields. Hash
    # the original payload rather than silently upgrading it during reads.
    request["version"] = 1
    payload = {k: v for k, v in request.items() if k != "request_id"}
    request["request_id"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    assert validate_request(request) == (w, d)
