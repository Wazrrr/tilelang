"""Kernel-specific child processes preserve workloads and comparison semantics."""

import importlib
import json
import os
from types import SimpleNamespace

import pytest

from experiments._common import prepare_run, write_json


CASES = [("gemm_fp8", ["--m", "256", "--n", "512", "--k", "128", "--dtype", dtype]) for dtype in ("float8_e4m3fn", "float8_e5m2")] + [
    ("flash_attention", ["--batch", "2", "--heads", "4", "--sequence", "256", "--dim", "64", *causal]) for causal in ([], ["--causal"])
]


@pytest.mark.parametrize("family,workload", CASES)
def test_system_children_keep_kernel_and_workload(tmp_path, monkeypatch, family, workload):
    runner = importlib.import_module(f"experiments.{family}.system.run")
    monkeypatch.setattr(os, "environ", os.environ.copy())
    args = runner.parse_args(
        [
            *workload,
            "--output",
            str(tmp_path),
            "--variant",
            "all",
            "--benchmark-devices",
            "0",
            "1",
            "--config-indices",
            "0",
            "8",
            "--workers",
            "2",
            "--seed",
            "456",
        ]
    )
    prepare_run(args)
    children = []

    def child_run(command, *, check):
        assert check
        assert command[1:3] == ["-m", f"experiments.{family}.system.run"]
        child = runner.parse_args(command[3:])
        for name, value in vars(args).items():
            if name not in ("output", "run_name", "variant"):
                assert getattr(child, name) == value
        prepare_run(child)
        assert child.output == args.output / child.variant
        write_json(child.output / "summary.json", dict(variant=child.variant, tuning_seconds=2, winner_latency_ms=0.1))
        children.append(child.variant)

    monkeypatch.setattr(runner.subprocess, "run", child_run)
    runner.run_all(args)
    assert children == ["baseline", "pipeline", "grouped", "multi_gpu", "combined"]
    assert len(json.loads((args.output / "comparison.json").read_text())) == 5


@pytest.mark.parametrize("family,workload", CASES)
@pytest.mark.parametrize("budget", ["1", "all"])
def test_comparison_children_preserve_budget_and_validation(tmp_path, monkeypatch, family, workload, budget):
    runner = importlib.import_module(f"experiments.{family}.tiletune.run")
    monkeypatch.setattr(os, "environ", os.environ.copy())
    args = runner.parse_args(
        [
            *workload,
            "--method",
            "all",
            "--top-k",
            budget,
            "--output",
            str(tmp_path),
            "--config-indices",
            "8",
            "16",
            "--group-size",
            "2",
            "--validation-repeats",
            "3",
            "--device-profile",
            str(tmp_path / "profile.json"),
        ]
    )
    prepare_run(args)
    children = []

    def child_run(command, *, stdout, stderr):
        assert command[1:3] == ["-m", f"experiments.{family}.tiletune.run"]
        child = runner.parse_args(command[3:])
        for name, value in vars(args).items():
            if name not in ("output", "run_name", "method"):
                assert getattr(child, name) == value
        prepare_run(child)
        selected = [0] if child.method == "tiletune" and budget == "1" else [0, 1]
        records = [
            dict(index=i, original_index=original, selected=i in selected, status="benchmarked", latency_ms=latency)
            for i, original, latency in [(0, 8, 2), (1, 16, 1)]
        ]
        write_json(child.output / "outcomes.json", records)
        if child.method == "tiletune":
            write_json(
                child.output / "tiletune.json",
                dict(
                    configs=records,
                    selection=dict(selected_indices=selected) if budget == "1" else None,
                    ranking=[
                        dict(index=i, rank=i + 1, score=i + 1, tier="eligible", tie_first_rank=i + 1, tie_last_rank=i + 1) for i in range(2)
                    ],
                ),
            )
        winner = selected[-1]
        write_json(
            child.output / "summary.json",
            dict(
                method=child.method,
                status="ok",
                selected_count=len(selected),
                successful_count=len(selected),
                tuning_seconds=1,
                winner=dict(index=winner, original_index=records[winner]["original_index"], predicted_rank=winner + 1),
            ),
        )
        children.append(child.method)
        return SimpleNamespace(returncode=0)

    def validate(parent_args, summaries):
        assert parent_args.validation_repeats == 3
        for row in summaries:
            row["validation_latency_ms"] = 1

    monkeypatch.setattr(runner.subprocess, "run", child_run)
    monkeypatch.setattr(runner, "remeasure_winners", validate)
    assert runner.run_all(args)
    assert children == ["tiletune", "brute_force"]
    comparison = json.loads((args.output / "comparison.json").read_text())
    tiletune = comparison["methods"][1]
    assert tiletune["top_k_oracle_retained_performance"] == (0.5 if budget == "1" else 1)
    assert tiletune["brute_force_winner_model_rank"]["selected"] == (budget == "all")


@pytest.mark.parametrize("family", ["gemm_fp8", "flash_attention"])
def test_oracle_excludes_failed_measurements_without_changing_selection(tmp_path, family):
    runner = importlib.import_module(f"experiments.{family}.tiletune.run")
    outputs = {method: tmp_path / method for method in runner.METHODS}
    for path in outputs.values():
        path.mkdir()
    write_json(
        outputs["brute_force"] / "outcomes.json",
        [
            dict(index=0, status="benchmarked", latency_ms=1),
            dict(index=1, status="compilation_failed"),
            dict(index=2, status="benchmarked", latency_ms=2),
        ],
    )
    report = dict(
        selection=dict(selected_indices=[1, 2]),
        configs=[dict(index=i, selected=i in [1, 2]) for i in range(3)],
        ranking=[
            dict(index=i, rank=rank, score=rank, tier="eligible", tie_first_rank=rank, tie_last_rank=rank)
            for rank, i in enumerate([1, 2, 0], 1)
        ],
    )
    path = outputs["tiletune"] / "tiletune.json"
    write_json(path, report)
    before = path.read_bytes()
    rows = [
        dict(method=m, winner=dict(index=i), validation_latency_ms=1, tuning_seconds=1) for m, i in [("brute_force", 0), ("tiletune", 2)]
    ]
    runner.compare_results(rows, outputs)
    assert rows[1]["selected_candidates_with_oracle_measurement"] == 1
    assert rows[1]["top_k_oracle_retained_performance"] == 0.5
    assert path.read_bytes() == before


def test_attention_flops_count_causal_pairs():
    from experiments.flash_attention.tiletune.run import workload_flops

    args = SimpleNamespace(batch=1, heads=1, dim=1, sequence=3, causal=False)
    assert workload_flops(args) == 36
    args.causal = True
    assert workload_flops(args) == 24
