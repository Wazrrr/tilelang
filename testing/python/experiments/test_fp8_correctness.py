"""FP8 rounding tolerance must reject corrupted outputs and contract changes."""

import pytest
import torch
from experiments.gemm_fp8.cases import cases
from experiments.gemm_fp8.kernel import make_case


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_fp8_check_accepts_one_sparse_rounding_step_and_rejects_corruption(dtype):
    case = make_case(cases()[0])
    expected = torch.ones(256).to(dtype)
    actual = expected.float()
    actual[0] += 0.125 if dtype == torch.float8_e4m3fn else 0.25
    case.check([actual.to(dtype)], [expected])
    for wrong in (torch.zeros(256).to(dtype), (actual + 1).to(dtype), actual, expected[:128]):
        with pytest.raises(AssertionError):
            case.check([wrong], [expected])
    actual[0] = float("nan")
    with pytest.raises(AssertionError):
        case.check([actual.to(dtype)], [expected])
    # Two steps below a power-of-two boundary are smaller than one upward
    # spacing; compare representable codes, not a maximum-magnitude ULP.
    actual = expected.float()
    actual[0] = 0.875 if dtype == torch.float8_e4m3fn else 0.75
    with pytest.raises(AssertionError):
        case.check([actual.to(dtype)], [expected])
