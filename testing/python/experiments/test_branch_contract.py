"""Shared contracts and mathematical checks for the three CUDA worktrees."""

from types import SimpleNamespace

import pytest

from experiments.common.spec import Workload
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


def test_carver_counts_fp8_scales(monkeypatch):
    from experiments.common import carver
    from experiments.gemm_fp8.carver import carver_rank as fp8_rank
    from tilelang import tvm

    arch = SimpleNamespace(
        target=tvm.target.Target(dict(kind="cuda", arch="sm_90")),
        max_smem_usage=227328,
        reg_cap=65536,
        sm_partition=4,
        compute_max_core=132,
    )
    monkeypatch.setattr(carver, "_architecture", lambda target: arch)
    device = SimpleNamespace(target=dict(kind="cuda", arch="sm_90a"))
    w = Workload("fp8", "gemm_fp8", dict(m=128, n=256, k=512, transpose_b=True), "float8_e4m3fn")
    config = dict(block_M=64, block_N=128, block_K=128, num_stages=1, threads=128)
    r = fp8_rank(w, device, [config], 1)["configs"][0]["model"]
    assert r["scale_bytes"] == 4 * (64 + 128) * 4
    assert r["register_words"] == 2 * 64 * 128
    assert r["traffic_bytes_per_cta"] == (64 + 128) * 512 + r["scale_bytes"] + 2 * 64 * 128


@pytest.mark.parametrize("m", [32, 128])
@pytest.mark.parametrize("compute_dtype", ["bfloat16", "float8_e4m3fn"])
def test_fp8_native_and_emulated_paths_share_the_scale_contract_on_gpu(compute_dtype, m):
    import torch
    import tilelang
    from tilelang.tiletune import current_target
    from examples.gemm_fp8.example_blockscaled_gemm import blockscaled_gemm, quantize_e4m3
    from experiments.gemm_fp8.reference import reference
    from experiments.utils.kernel import KernelCase

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    major, minor = torch.cuda.get_device_capability()
    if compute_dtype == "float8_e4m3fn" and (major, minor) < (8, 9):
        pytest.skip("native FP8 requires SM89 or newer")
    torch.backends.cuda.matmul.allow_tf32 = False
    g = torch.Generator(device="cuda").manual_seed(123)
    a, sa = quantize_e4m3(torch.randn(m, 256, device="cuda", generator=g).bfloat16())
    b, sb = quantize_e4m3(torch.randn(256, 256, device="cuda", generator=g).bfloat16())
    # Distinct scales on adjacent rows detect accidental 128-row sharing.
    sa *= torch.linspace(0.5, 1.5, m, device="cuda").unsqueeze(1)
    sb *= torch.linspace(0.5, 1.5, 256, device="cuda").unsqueeze(1)
    func = blockscaled_gemm.get_tir(M=m, N=256, K=256, block_M=64, block_N=64, num_stages=1, threads=128, compute_dtype=compute_dtype)
    compiled = tilelang.compile(func, target=current_target(), execution_backend="tvm_ffi")
    KernelCase(None, None, None, None, rtol=0.03, atol=0.03).check([compiled(a, b, sa, sb)], [reference(a, b, sa, sb)])
