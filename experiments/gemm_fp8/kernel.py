"""Use the authoritative FP8 example, including its FP8 output cast."""

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
            if actual.dtype != expected.dtype or actual.shape != expected.shape:
                raise AssertionError("FP8 output shape or dtype differs from the reference")
            a, b = actual.float(), expected.float()

            # FP32 reduction order can put a value on either side of an FP8
            # rounding midpoint. Permit one adjacent representable value,
            # including subnormals, while retaining the 2% global norm check.
            # Magnitude encodings are monotonic in both supported formats.
            # Signed integer distances handle the asymmetric spacing at powers
            # of two and identify positive/negative zero as the same value.
            def ordered_bits(tensor):
                bits = tensor.view(torch.uint8).to(torch.int16)
                magnitude = bits & 127
                return torch.where(bits & 128 != 0, -magnitude, magnitude)

            adjacent = (ordered_bits(actual) - ordered_bits(expected)).abs() <= 1
            error = (a - b).abs()
            if not torch.isfinite(a).all() or not torch.isfinite(b).all() or not adjacent.all():
                raise AssertionError("FP8 output differs by more than one quantization step")
            relative = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(b).clamp_min(1e-12)
            if relative > self.rtol:
                raise AssertionError(f"relative FP8 output norm error {relative.item()} exceeds {self.rtol}")


def make_case(workload):
    reason = support_reason(workload)
    if reason:
        raise ValueError(reason)
    p, dtype = workload.parameters, workload.dtype
    m, n, k = p["m"], p["n"], p["k"]

    def build(block_M, block_N, block_K, num_stages, threads, enable_rasteration):
        from examples.gemm_fp8.example_tilelang_gemm_fp8 import matmul

        with _BUILD_LOCK:
            return matmul.get_tir(
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

    # The eager example declares C with T.empty and embeds its output index.
    return FP8KernelCase(build, inputs, reference, None)
