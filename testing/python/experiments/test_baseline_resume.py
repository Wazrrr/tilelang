"""Keep complete failed shortlists while rejecting interrupted measurements."""
import json
from pathlib import Path
import subprocess

import pytest

from experiments.utils.baseline_store import collect_bundle, exhausted_selection
from experiments.common.spec import Device, TARGETS


@pytest.mark.parametrize("status,monitor,expected", [
    ("compilation_failed", "uncontended", True),
    ("benchmark_error", "uncontended", True),
    ("compiled", "uncontended", False),
    ("worker_failed", "uncontended", False),
    ("benchmark_error", "contended", False),
])
def test_failed_shortlist_requires_terminal_uncontended_records(tmp_path, status, monitor, expected):
    (tmp_path / "result.json").write_text(json.dumps(dict(status="failed")))
    (tmp_path / "monitor.json").write_text(json.dumps(dict(status=monitor)))
    (tmp_path / "carver.json").write_text(json.dumps(dict(
        selection=dict(selected_indices=[0, 1]),
        configs=[dict(index=i, status=status) for i in range(2)],
        ranking=[dict(index=i) for i in range(2)],
    )))
    assert exhausted_selection(tmp_path, "carver") is expected


def test_partial_bundle_resumes_without_discarding_saved_measurements(tmp_path):
    identity = dict(splits=dict(test=[]), baseline_seed=123)
    calls = []

    def collect(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        calls.append(command)
        if len(calls) == 1:
            output.mkdir()
            (output / "saved.json").write_text('{"measurement": 42}')
            raise subprocess.CalledProcessError(2, command)
        assert "--resume" in command
        assert (output / "saved.json").read_text() == '{"measurement": 42}'

    device = Device("ampere", TARGETS["ampere"])
    with pytest.raises(subprocess.CalledProcessError):
        collect_bundle(tmp_path, identity, {}, device, {}, run=collect)
    path, reused = collect_bundle(tmp_path, identity, {}, device, {}, run=collect)
    assert not reused and (path / "complete.json").exists()
    assert collect_bundle(tmp_path, identity, {}, device, {}, run=collect) == (path, True)
    assert len(calls) == 2
