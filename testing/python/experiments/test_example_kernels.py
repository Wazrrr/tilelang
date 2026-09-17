"""Final workloads must elaborate the example programs, not parallel rewrites."""

from dataclasses import replace

import pytest

from experiments.suite import core_cases
from experiments.common.spec import Device, TARGETS, configurations

EXAMPLE_CONFIGS = {
    "gemm": dict(block_M=128, block_N=256, block_K=64, num_stages=3, thread_num=256, enable_rasteration=True),
    "attention": dict(block_M=64, block_N=64, num_stages=1, threads=128),
    "kda_chunk_o": dict(block_DK=64, block_DV=64, num_stages=0, threads=128),
    "gemm_fp8": dict(block_M=128, block_N=128, block_K=64, num_stages=3, threads=128, enable_rasteration=False),
}


def example_program(w, c):
    """Build through the example's public API independently of suite dispatch."""
    p = w.parameters
    if w.op == "gemm":
        from examples.gemm.example_gemm_advanced_autotune import make_autotune_kernel_builder

        return make_autotune_kernel_builder(p["m"], p["n"], p["k"], w.dtype)(**c)
    if w.op == "attention":
        from examples.flash_attention.example_mha_fwd_bshd import flashattn

        return flashattn.jit_impl.get_tir(
            batch=p["batch"],
            heads=p["heads"],
            seq_len=p["sequence"],
            dim=p["dim"],
            is_causal=p["causal"],
            dtype=w.dtype,
            **c,
        )
    if w.op == "kda_chunk_o":
        from examples.kda.chunk_o import tilelang_chunk_fwd_o

        return tilelang_chunk_fwd_o.jit_impl.get_tir(
            B=p["batch"],
            S=p["sequence"],
            H=p["heads"],
            DK=p["dim"],
            DV=p["value_dim"],
            input_dtype=w.dtype,
            output_dtype=w.dtype,
            accum_dtype="float32",
            gate_dtype="float32",
            chunk_size=p["chunk_size"],
            scale=p["dim"] ** -0.5,
            block_S=p["chunk_size"],
            block_DK=c["block_DK"],
            block_DV=c["block_DV"],
            threads=c["threads"],
            num_stages=c["num_stages"],
        )
    from examples.gemm_fp8.example_tilelang_gemm_fp8 import matmul

    return matmul.get_tir(M=p["m"], N=p["n"], K=p["k"], dtype=w.dtype, **c)


@pytest.mark.parametrize("w", core_cases("final"), ids=lambda w: w.name)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_final_programs_are_structurally_identical_to_examples(w, dtype):
    from tilelang import tvm
    from experiments.common.kernels import make_case

    w = replace(w, dtype=("float8_e4m3fn" if dtype == "float16" else "float8_e5m2") if w.op == "gemm_fp8" else dtype)
    case = make_case(w)
    c = EXAMPLE_CONFIGS[w.op]
    assert all(isinstance(cell.cell_contents, (int, float, str, bool, type(None))) for cell in case.build.__closure__ or [])
    tvm.ir.assert_structural_equal(case.build(**c), example_program(w, c))


@pytest.mark.parametrize("target", ["ampere", "hopper", "blackwell", "mi355x"])
def test_expanded_retains_every_advanced_example_configuration(target):
    from examples.gemm.example_gemm_advanced_autotune import get_configs

    d = Device(target, TARGETS[target])
    for w in core_cases("final")[:2]:
        pool = configurations(w, d)
        assert len(pool) == 2304
        for c in get_configs(w.parameters["m"], w.parameters["n"], w.parameters["k"]):
            assert c in pool


@pytest.mark.parametrize("w", core_cases("final"), ids=lambda w: w.name)
def test_final_example_kernels_on_gpu(w):
    import torch
    import tilelang
    from experiments.common.kernels import make_case
    from tilelang.tiletune import current_target

    if not torch.cuda.is_available():
        pytest.skip("CUDA or ROCm required")
    from experiments.common.spec import support_reason

    reason = support_reason(w, Device("test", current_target()))
    if reason:
        pytest.skip(reason)
    case = make_case(w)
    c = EXAMPLE_CONFIGS[w.op]
    kernel = tilelang.compile(
        case.build(**c), target=current_target(), execution_backend="tvm_ffi", out_idx=case.out_idx, pass_configs=case.pass_configs
    )
    inputs = case.inputs("cuda", torch.Generator(device="cuda").manual_seed(123))
    case.check([kernel(*inputs)], [case.reference(*inputs)])
