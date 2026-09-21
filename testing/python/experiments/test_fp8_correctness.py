"""The FP8 adapter preserves the original two-input, FP8-output contract."""

import pytest
import torch
from experiments.gemm_fp8.cases import cases
from experiments.gemm_fp8.kernel import make_case


def test_fp8_inputs_and_reference_match_the_original_example_contract():
    case = make_case(cases()[0])
    inputs = case.inputs("cpu", torch.Generator().manual_seed(123))
    a, b = inputs
    assert a.dtype == b.dtype == torch.float8_e4m3fn
    assert a.shape[1] == b.shape[1]
    expected = case.reference(*inputs)
    assert expected.dtype == torch.float8_e4m3fn
    assert expected.shape == (a.shape[0], b.shape[0])
    case.check([expected], [expected])
    with pytest.raises(AssertionError):
        case.check([torch.zeros_like(expected)], [expected])
    for wrong in (expected.float(), expected[:, : expected.shape[1] // 2]):
        with pytest.raises(AssertionError):
            case.check([wrong], [expected])
