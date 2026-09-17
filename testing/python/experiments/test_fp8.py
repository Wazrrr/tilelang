"""Ampere's explicit-scale FP8-storage/BF16-compute contract."""

from dataclasses import replace

import pytest

from experiments.common.spec import Device, TARGETS, support_reason
from experiments.gemm_fp8.cases import cases


def test_ampere_fp8_contract_and_scale_layout():
    import torch
    from experiments.gemm_fp8.kernel import make_case

    w = replace(cases()[0], parameters=dict(m=128, n=128, k=128, transpose_b=True))
    assert support_reason(w, Device("ampere", TARGETS["ampere"])) is None
    case = make_case(w)
    a, b, scale_a, scale_b = case.inputs("cpu", torch.Generator().manual_seed(123))
    assert a.dtype == b.dtype == torch.float8_e4m3fn
    assert scale_a.shape == scale_b.shape == (128, 1)
    expected = case.reference(a, b, scale_a, scale_b)
    assert expected.dtype == torch.bfloat16
    case.check([expected], [expected])
    for wrong in (expected.float(), torch.zeros_like(expected), expected[:, :64]):
        with pytest.raises(AssertionError):
            case.check([wrong], [expected])


def test_e5m2_and_unaligned_shapes_are_outside_the_fixed_contract():
    from experiments.gemm_fp8.kernel import make_case

    with pytest.raises(ValueError, match="E4M3"):
        make_case(replace(cases()[0], dtype="float8_e5m2"))
    with pytest.raises(ValueError, match="M%32"):
        make_case(replace(cases()[0], parameters=dict(m=97, n=128, k=128, transpose_b=True)))


def test_adapter_program_is_the_ampere_example():
    from tilelang import tvm
    from examples.gemm_fp8.example_mxfp8_blockscaled_gemm_a100 import blockscaled_gemm
    from experiments.gemm_fp8.kernel import make_case

    w = replace(cases()[0], parameters=dict(m=128, n=128, k=128, transpose_b=True))
    config = dict(block_M=64, block_N=64, block_K=128, num_stages=1, threads=128)
    expected = blockscaled_gemm.get_tir(M=128, N=128, K=128, block_M=64, block_N=64, num_stages=1, threads=128)
    tvm.ir.assert_structural_equal(make_case(w).build(**config), expected)


def test_ampere_emulation_cross_compiles_for_sm80():
    import tilelang
    from experiments.common.spec import Workload
    from experiments.gemm_fp8.kernel import make_case

    w = Workload("compile", "gemm_fp8", dict(m=128, n=128, k=128, transpose_b=True), dtype="float8_e4m3fn")
    case = make_case(w)
    program = case.build(block_M=64, block_N=64, block_K=128, num_stages=1, threads=128)
    kernel = tilelang.compile(program, target=TARGETS["ampere"], execution_backend="tvm_ffi", out_idx=case.out_idx)
    assert kernel is not None
