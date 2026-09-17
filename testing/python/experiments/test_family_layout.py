"""Family entry points share the frozen protocol and preserve old artifacts."""

import importlib
from dataclasses import make_dataclass
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.common.acceptance import aggregate_study
from experiments.common.spec import Device, TARGETS
from experiments.suite import CORE_FAMILIES, core_cases, study_plan


def test_standard_studies_support_runtimes_without_optional_exploration():
    from experiments.common.run import exploration_options

    old_config = make_dataclass("OldConfig", [])
    assert exploration_options({"seed": 456}, old_config) == {}
    with pytest.raises(ValueError, match="runtime with exploration support"):
        exploration_options({"exploration_fraction": 0.2}, old_config)
    new_config = make_dataclass("NewConfig", [("exploration_fraction", float), ("exploration_seed", int)])
    assert exploration_options({"exploration_fraction": 0.2, "seed": 456, "selection_seed": 789}, new_config) == dict(
        exploration_fraction=0.2, exploration_seed=789
    )


@pytest.mark.parametrize("family", CORE_FAMILIES)
def test_family_command_plans_without_a_compiler_or_runtime(family):
    root = str(Path(__file__).resolve().parents[3])
    code = f"""
import sys
sys.path.insert(0, {root!r})
from experiments.{family}.tiletune.run import main
assert main(['--suite', 'smoke', '--device', 'ampere', '--config-space', {"expanded"!r}, '--plan']) == 0
assert not any(name.split('.')[0] in ('tilelang', 'torch', 'tvm', 'xgboost') for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-I", "-S", "-c", code], capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["families"] == [family]
    assert len(plan["splits"]["test"]) == 1
    assert not plan["splits"]["train"]
    assert len(plan["subsets"]["ampere"]) == 1


@pytest.mark.parametrize("family", CORE_FAMILIES)
def test_family_census_uses_its_declared_cases(family, capsys):
    runner = importlib.import_module(f"experiments.{family}.census")
    assert runner.main(["--config-space", "expanded", "--plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert {c["workload"]["name"] for c in plan["cases"]} == {w.name for w in core_cases("development", [family])}
    assert all(c["indices"] == list(range(c["space"]["candidate_count"])) for c in plan["cases"])


def test_named_suite_declares_three_baselines_and_independent_tiletune_repeats():
    plan = study_plan("full", [Device("hopper", TARGETS["hopper"])])
    assert plan["methods"] == ["brute_force", "carver", "xgboost", "tiletune"]
    assert plan["budget"]["seeds"] == [123, 456, 789]


def good_comparison(case):
    return dict(
        workload=case,
        methods=dict(
            tiletune=dict(status="completed", correctness="passed", tuning_seconds=1),
            brute_force=dict(tuning_seconds=2),
        ),
        diagnostics=dict(tiletune=dict(correct_score_coverage=0.95, curves={"20": dict(oracle_at_k=0.98)})),
        validation=dict(tiletune=dict(samples_ms=[1] * 7)),
    )


def test_family_acceptance_has_local_costs_and_cannot_certify_the_matrix(tmp_path):
    cases = [w.to_dict() for w in core_cases("final", ["gemm"])]
    plan = dict(suite="final", devices=[dict(name="ampere")], budget=dict(seeds=[123]), splits=dict(test=cases), unavailable={})
    result = tmp_path / "123/ampere/comparison.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps(dict(results=[good_comparison(c) for c in cases])))
    profile = tmp_path / "preparation/ampere/result.json"
    profile.parent.mkdir(parents=True)
    profile.write_text(json.dumps(dict(preparation_seconds=2, status="profiled")))
    report = aggregate_study(plan, tmp_path)
    assert report["accepted"] and report["scope"] == "families"
    assert not report["five_target_accepted"]
    costs = report["targets"]["ampere"]["seeds"]["123"]["costs"]["tiletune"]
    assert costs["online_seconds"] == 5
    assert costs["amortized_seconds_per_case"] == 1.4
    # The complete case set still requires all five devices for final acceptance.
    plan["splits"]["test"] = [w.to_dict() for w in core_cases("final")]
    result.write_text(json.dumps(dict(results=[good_comparison(c) for c in plan["splits"]["test"]])))
    assert not aggregate_study(plan, tmp_path)["accepted"]


def test_xgboost_fingerprints_family_implementation_sources():
    from experiments.utils.cli import source_hashes
    from experiments.xgboost.data import domain, make_context

    w = core_cases("final", ["gemm"])[0]
    sources = source_hashes("experiments/common/kernels.py")
    args = (w, "portable.gemm", TARGETS["ampere"], "A100", "event")
    before = make_context(*args, sources)
    example = "examples/gemm/example_gemm_advanced_autotune.py"
    assert example in before["kernel_sha256"]
    sources[example] = "changed example implementation"
    assert domain(before) != domain(make_context(*args, sources))
    before = make_context(*args, sources)
    sources["experiments/gemm/kernel.py"] = "changed implementation"
    assert domain(before) != domain(make_context(*args, sources))
    del sources["experiments/gemm/reference.py"]
    with pytest.raises(ValueError, match="missing kernel source fingerprints"):
        make_context(*args, sources)


def test_audit_command_plans_without_runtime():
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "-S", "-m", "experiments.common.audit_model", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--output" in result.stdout
