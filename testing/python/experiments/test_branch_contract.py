"""Shared contracts and mathematical checks for the three CUDA worktrees."""

from types import SimpleNamespace

from experiments.families import FAMILIES, family_module


def test_every_shape_split_is_disjoint_and_has_the_declared_dtype():
    from experiments.xgboost.data import canonical_workload, digest

    for method in ("development", "final", "training_validation"):

        def shapes(op, method):
            mod = family_module(op, "cases")
            workloads = mod.training_cases() if method == "training_validation" else mod.cases(holdout=method == "final")
            return [w.parameters for w in workloads]

        assert shapes("gemm", method) == shapes("gemm_fp8", method)
    for op in FAMILIES:
        mod = family_module(op, "cases")
        training = mod.training_cases()
        final = mod.cases(holdout=True)
        assert len(training) == 3 and len(final) == 5
        assert len({digest(canonical_workload(w)) for w in training + final}) == 8
        assert {w.dtype for w in training + final} == {"float8_e4m3fn" if op == "gemm_fp8" else "bfloat16"}
        if op.startswith("kda_"):
            assert all(tuple(w.parameters[k] for k in ("dim", "chunk_size", "sub_chunk_size")) == (128, 64, 16) for w in final)


def test_fp8_uses_original_grid_and_output_contract():
    import torch
    from examples.gemm_fp8.example_gemm_fp8_tiletune import get_configs
    from experiments.gemm_fp8.spaces import get_configs as expanded
    from experiments.gemm_fp8.reference import reference

    assert all(c in expanded() for c in get_configs())
    a = torch.tensor([[1.0, 2.0]], dtype=torch.float8_e4m3fn)
    b = torch.tensor([[3.0, 4.0]], dtype=torch.float8_e4m3fn)
    actual = reference(a, b)
    assert actual.dtype == a.dtype
    assert actual.shape == (1, 1)
    assert actual.float().item() == 11.0


def test_fp8_carver_is_explicitly_deferred_for_changed_contract():
    from experiments.common.baselines import carver_support_reason

    w = family_module("gemm_fp8", "cases").cases(holdout=True)[0]
    assert "deferred" in carver_support_reason(w, SimpleNamespace(target=dict(kind="cuda", arch="sm_90a")))
