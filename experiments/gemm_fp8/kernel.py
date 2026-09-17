"""Use the authoritative FP8 example with FP32 accumulation and FP8 output."""

from threading import Lock
import torch

from experiments.utils.kernel import KernelCase, _random
from .reference import reference
from .spaces import support_reason

_BUILD_LOCK = Lock()


class FP8KernelCase(KernelCase):
    def check(self, actuals, references):
        if len(actuals) != len(references):
            raise AssertionError("wrong number of kernel outputs")
        for actual, expected in zip(actuals, references):
            if (
                actual.shape != expected.shape
                or actual.dtype != expected.dtype
                or expected.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
            ):
                raise AssertionError("FP8 output shape/dtype differs from the reference")
            # Independent accumulation can cross an FP8 rounding boundary.
            # Allow at most one representable step, plus the example's global
            # relative-energy criterion; zero/NaN outputs cannot pass.
            x, y = actual.float(), expected.float()

            def ordered_codes(value):
                bits = value.contiguous().view(torch.uint8).to(torch.int16)
                magnitude = bits & 127
                return torch.where(bits & 128 != 0, -magnitude, magnitude)

            if (
                not torch.isfinite(x).all()
                or not torch.isfinite(y).all()
                or not torch.all((ordered_codes(actual) - ordered_codes(expected)).abs() <= 1)
            ):
                raise AssertionError("FP8 output differs by more than one representable step")
            energy = (x.double().square() + y.double().square()).sum()
            difference = (x.double() - y.double()).square().sum() / energy.clamp_min(1e-30)
            if not torch.isfinite(difference) or difference >= 1e-3:
                raise AssertionError(f"FP8 relative energy error {difference.item()} exceeds 1e-3")


def _program(**kwargs):
    from examples.gemm_fp8.example_tilelang_gemm_fp8 import matmul

    with _BUILD_LOCK:
        return matmul.get_tir(**kwargs)


def make_case(workload):
    reason = support_reason(workload)
    if reason:
        raise ValueError(reason)
    m, n, k = (workload.parameters[key] for key in ("m", "n", "k"))
    dtype = workload.dtype

    def build(block_M, block_N, block_K, num_stages, threads, enable_rasteration):
        return _program(
            M=m,
            N=n,
            K=k,
            dtype=dtype,
            block_M=block_M,
            block_N=block_N,
            block_K=block_K,
            num_stages=num_stages,
            threads=threads,
            enable_rasteration=enable_rasteration,
        )

    def inputs(device, generator):
        return [_random(shape, dtype, device, generator) for shape in ((m, k), (n, k))]

    return FP8KernelCase(build, inputs, reference, None)
