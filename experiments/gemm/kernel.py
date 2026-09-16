"""Inputs and numerical reference for the advanced example's GEMM kernel."""

from examples.gemm.example_gemm_advanced_autotune import make_autotune_kernel_builder
from experiments.utils.kernel import KernelCase, _random
from .reference import reference
from .spaces import support_reason


def make_case(workload):
    reason = support_reason(workload)
    if reason:
        raise ValueError(reason)
    p, dtype = workload.parameters, workload.dtype
    m, n, k = p["m"], p["n"], p["k"]

    def inputs(device, generator):
        return [_random(shape, dtype, device, generator) for shape in ((m, k), (n, k))]

    return KernelCase(make_autotune_kernel_builder(m, n, k, dtype), inputs, reference, [2])
