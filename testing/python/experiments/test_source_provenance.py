"""Generated artifacts must not alter active experiment source identities."""

from experiments.utils import cli as _common


def test_archived_python_is_excluded_while_active_kernel_edits_change_hashes(tmp_path, monkeypatch):
    active = tmp_path / "experiments/gemm/kernel.py"
    active.parent.mkdir(parents=True)
    active.write_text("kernel = 1\n")
    helper = tmp_path / "experiments/utils/cli.py"
    helper.parent.mkdir()
    helper.write_text("# provenance code\n")
    monkeypatch.setattr(_common, "__file__", str(helper))
    before = _common.source_hashes("experiments/gemm/kernel.py")
    archived = tmp_path / "experiments/results/run/sources/kernel.py"
    archived.parent.mkdir(parents=True)
    archived.write_text("old_kernel = 1\n")
    assert _common.source_hashes("experiments/gemm/kernel.py") == before
    archived.write_text("old_kernel = 2\n")
    assert _common.source_hashes("experiments/gemm/kernel.py") == before
    active.write_text("kernel = 2\n")
    after = _common.source_hashes("experiments/gemm/kernel.py")
    assert before["experiments/gemm/kernel.py"] != after["experiments/gemm/kernel.py"]
    assert before["experiments/utils/cli.py"] == after["experiments/utils/cli.py"]
