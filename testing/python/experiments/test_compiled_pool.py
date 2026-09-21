"""Changed inputs must not inherit a prior pool's compilation qualification."""

import hashlib
import json

import pytest

from experiments.utils import compiled_pool


def test_compiled_pool_keeps_only_confirmed_configs_and_rejects_changed_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(compiled_pool, "ROOT", tmp_path)
    source = tmp_path / "kernel.py"
    source.write_text("original kernel")
    candidates = [{"tile": 32}, {"tile": 96}, {"tile": 128}]
    directory = tmp_path / "experiments/compilation"
    directory.mkdir(parents=True)
    (directory / "gemm.json").write_text(
        json.dumps(
            dict(
                candidate_pool_sha256=compiled_pool.pool_digest(candidates),
                candidate_count=3,
                compiled_count=2,
                accepted_indices=[0, 2],
                kernel_sources={"kernel.py": hashlib.sha256(source.read_bytes()).hexdigest()},
            )
        )
    )
    assert compiled_pool.compiled_configs("gemm", candidates) == [candidates[0], candidates[2]]
    with pytest.raises(ValueError, match="candidate grid changed"):
        compiled_pool.compiled_configs("gemm", candidates + [{"tile": 256}])
    source.write_text("changed kernel")
    with pytest.raises(ValueError, match="compilation source changed"):
        compiled_pool.compiled_configs("gemm", candidates)
