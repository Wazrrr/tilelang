"""Repeated experiments preserve old data and keep comparisons in one run."""

import argparse
import json
import os

import pytest

from experiments._common import add_run_arguments, prepare_run, write_json
from experiments.gemm.system import run as system


def arguments(output, *extra):
    parser = argparse.ArgumentParser()
    add_run_arguments(parser, output)
    return parser.parse_args(list(extra))


def test_repeated_runs_preserve_existing_results(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    old_summary = tmp_path / "summary.json"
    old_summary.write_text('"original"\n')
    first = arguments(tmp_path)
    prepare_run(first)
    write_json(first.output / "summary.json", "first run")
    second = arguments(tmp_path)
    prepare_run(second)
    assert first.output != second.output
    assert first.output.parent == second.output.parent == tmp_path
    assert old_summary.read_text() == '"original"\n'
    assert json.loads((first.output / "summary.json").read_text()) == "first run"
    assert not (second.output / "summary.json").exists()
    assert os.environ["TILELANG_AUTOTUNE_TIMING_LOG"] == str(second.output / "timings.tsv")
    assert os.environ["TILELANG_DISABLE_CACHE"] == os.environ["TILELANG_AUTO_TUNING_DISABLE_CACHE"] == "1"


@pytest.mark.parametrize("populated", [False, True])
def test_named_run_cannot_reuse_an_existing_directory(tmp_path, monkeypatch, populated):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    first = arguments(tmp_path, "--run-name", "v2")
    prepare_run(first)
    assert first.output == tmp_path / "v2"
    if populated:
        write_json(first.output / "summary.json", "saved")
    with pytest.raises(ValueError, match="Run directory already exists"):
        prepare_run(arguments(tmp_path, "--run-name", "v2"))
    if populated:
        assert json.loads((first.output / "summary.json").read_text()) == "saved"


def test_all_variants_share_one_version_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    args = arguments(tmp_path)
    prepare_run(args)
    vars(args).update(m=256, n=256, k=256, dtype="bfloat16", group_size=2, backend="event", benchmark_devices=[0, 1])

    def write_child_results(command, *, check):
        # Exercise the child's real output parsing without compiling GPU kernels.
        parser = argparse.ArgumentParser()
        add_run_arguments(parser, tmp_path)
        parser.add_argument("--variant")
        child, _ = parser.parse_known_args(command[3:])
        prepare_run(child)
        write_json(child.output / "summary.json", dict(variant=child.variant, tuning_seconds=1.0, winner_latency_ms=0.1))

    monkeypatch.setattr(system.subprocess, "run", write_child_results)
    system.run_all(args)
    comparison = json.loads((args.output / "comparison.json").read_text())
    assert [row["variant"] for row in comparison] == list(system.VARIANTS)
    for variant in system.VARIANTS:
        assert (args.output / variant / "summary.json").is_file()
    assert list(tmp_path.iterdir()) == [args.output]
