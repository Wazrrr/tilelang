"""The study must finish all selections before starting any shared oracle."""

import hashlib
import json
from pathlib import Path

import pytest

from experiments.portable import repair_study


def test_all_seeds_freeze_before_oracles_and_resume_rejects_changed_rankings(tmp_path, monkeypatch):
    methods = ["tiletune", "tiletune_exploration", "xgboost", "xgboost_stratified", "random"]
    manifest = {"devices": [{"name": "ampere"}], "splits": {"test": [{"name": "heldout"}]}}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    root = tmp_path / "study"
    root.mkdir()
    (root / "splits-profiled.json").write_text(json.dumps(manifest))
    (root / "profile-preparation.json").write_text(json.dumps({"seconds": 3}))
    calls, selected = [], set()
    changed = False

    def run(command, **kwargs):
        phase = command[command.index("--phase") + 1]
        seed = int(command[command.index("--seed") + 1])
        output = Path(command[command.index("--output") + 1])
        calls.append((phase, seed))
        if phase == "selection":
            selected.add(seed)
            case = output / "ampere/test/heldout"
            case.mkdir(parents=True, exist_ok=True)
            (case / "methods.json").write_text(json.dumps({method: {"status": "completed"} for method in methods}))
            for method in methods:
                report_name = "tiletune" if method.startswith("tiletune") else "xgboost" if method.startswith("xgboost") else method
                folder = case / method
                folder.mkdir(exist_ok=True)
                report = {"ranking": [{"index": int(changed), "score": 1}], "selection": {"selected_indices": [int(changed)]}}
                (folder / (report_name + ".json")).write_text(json.dumps(report))
        else:
            assert selected == {123, 456, 789}
            frozen = json.loads((root / "all-selections-frozen.json").read_text())
            assert len(frozen["methods"]) == 3
            assert frozen["sha256"] == hashlib.sha256(json.dumps(frozen["methods"], sort_keys=True).encode()).hexdigest()
            assert command[command.index("--oracle-root") + 1] == str(root / "oracle")
            assert command[command.index("--validation-repeats") + 1] == "7"
            result = {"workload": {"op": "gemm"}, "validation": {method: {"performance_vs_brute_force": 0.95} for method in methods}}
            (output / "comparison.json").write_text(json.dumps({"results": [result]}))

    monkeypatch.setattr(repair_study.subprocess, "run", run)
    arguments = ["--manifest", str(manifest_path), "--output", str(root), "--resume"]
    assert repair_study.main(arguments) == 0
    assert calls == [(phase, seed) for phase in ("selection", "oracle") for seed in (123, 456, 789)]
    study = json.loads((root / "study.json").read_text())
    assert study["plan"]["top_k"] == 20
    assert study["plan"]["sample_fraction"] == 0.1
    assert all(quality["median"] == 0.95 for quality in study["oracle_at_20"].values())
    changed = True
    calls.clear()
    with pytest.raises(ValueError, match="selections changed after freezing"):
        repair_study.main(arguments)
    assert all(phase == "selection" for phase, _ in calls)


@pytest.mark.parametrize("seeds", [[123, 123], [-1]])
def test_study_rejects_duplicate_or_negative_seeds(tmp_path, seeds):
    with pytest.raises(SystemExit):
        repair_study.main(["--output", str(tmp_path / "study"), "--seeds", *map(str, seeds), "--plan"])
