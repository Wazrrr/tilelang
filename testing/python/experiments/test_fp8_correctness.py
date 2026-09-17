"""The FP8 adapter exposes explicit scales and a BF16 output contract."""

import pytest
import torch
from experiments.gemm_fp8.cases import cases
from experiments.gemm_fp8.kernel import make_case


def test_fp8_inputs_have_fixed_scale_layout_and_bf16_reference():
    case = make_case(cases()[0])
    inputs = case.inputs("cpu", torch.Generator().manual_seed(123))
    a, b, scale_a, scale_b = inputs
    assert a.dtype == b.dtype == torch.float8_e4m3fn
    assert scale_a.shape == (a.shape[0], a.shape[1] // 128)
    assert scale_b.shape == (b.shape[0] // 128, b.shape[1] // 128)
    expected = case.reference(*inputs)
    assert expected.dtype == torch.bfloat16
    case.check([expected], [expected])
    with pytest.raises(AssertionError):
        case.check([torch.zeros_like(expected)], [expected])
    for wrong in (expected.float(), expected[:, : expected.shape[1] // 2]):
        with pytest.raises(AssertionError):
            case.check([wrong], [expected])
