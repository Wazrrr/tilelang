"""The original H200 FP8 example, including its rasterization parameter."""

from threading import Lock

from experiments.utils.kernel import KernelCase, _random
from .reference import reference
from .spaces import support_reason

_BUILD_LOCK = Lock()


class FP8KernelCase(KernelCase):
    """Use the original example's normalized similarity check for FP8 output."""

    def check(self, actuals, references):
        if len(actuals) != len(references):
            raise AssertionError("wrong number of kernel outputs")
        for actual, reference_value in zip(actuals, references):
            if actual.dtype != reference_value.dtype:
                raise AssertionError(f"output dtype {actual.dtype} differs from reference {reference_value.dtype}")
            if actual.shape != reference_value.shape or actual.device != reference_value.device:
                raise AssertionError("output shape or device differs from reference")
            x, y = actual.double(), reference_value.double()
            denominator = (x * x + y * y).sum()
            difference = 1 - 2 * (x * y).sum() / denominator
            if not difference.isfinite() or difference >= 1e-3:
                raise AssertionError(f"FP8 example correctness failed: calc_diff={difference.item()}")


def make_case(workload):
    reason = support_reason(workload)
    if reason:
        raise ValueError(reason)
    m, n, k = (workload.parameters[key] for key in ("m", "n", "k"))
    dtype = workload.dtype

    def build(block_M, block_N, block_K, num_stages, threads, enable_rasteration):
        from examples.gemm_fp8.example_tilelang_gemm_fp8 import matmul

        with _BUILD_LOCK:
            return matmul.get_tir(
                M=m,
                N=n,
                K=k,
                block_M=block_M,
                block_N=block_N,
                block_K=block_K,
                dtype=dtype,
                num_stages=num_stages,
                threads=threads,
                enable_rasteration=enable_rasteration,
            )

    def inputs(device, generator):
        return [_random(shape, dtype, device, generator) for shape in ((m, k), (n, k))]

    return FP8KernelCase(build, inputs, reference, None, rtol=1e-3, atol=0)
