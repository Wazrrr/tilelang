"""Grouped compilation must honor eager example output attributes without TileTune."""

import pytest
import torch
import tilelang
from tilelang.autotuner.grouped_compile import compile_grouped_unit_tvm_ffi
from tilelang.autotuner.param import CompileArgs
from experiments.utils.kernel import KernelCase, _random


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_grouped_softmax_preserves_example_output_contract():
    from tilelang.tiletune import current_target

    from examples.online_softmax.online_softmax import softmax_kernel
    import tilelang.language as T

    def build(**config):
        return softmax_kernel.get_tir(T.Tensor((8, 128), "float16"), dtype="float16", **config)

    case = KernelCase(
        build, lambda device, gen: [_random((8, 128), "float16", device, gen)], lambda x: x.float().softmax(-1).to(x.dtype), None
    )
    target = tilelang.tvm.target.Target(current_target())
    configs = [dict(BLOCK_M=1, BLOCK_N=128, threads=t) for t in (64, 128)]
    before = [case.build(**c).script() for c in configs]
    results = compile_grouped_unit_tvm_ffi(
        list(enumerate(configs)), CompileArgs(target=target, out_idx=None, execution_backend="tvm_ffi"), case.build
    )
    inputs = case.inputs("cuda", torch.Generator(device="cuda").manual_seed(123))
    for _, _, kernel, error in results:
        assert error is None, error
        case.check([kernel(*inputs)], [case.reference(*inputs)])
    assert [case.build(**c).script() for c in configs] == before
