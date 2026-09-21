"""New schedule parameters must preserve tails, masking and reductions on GPU."""

import pytest

from experiments.common.spec import Workload


@pytest.fixture(scope="module")
def gpu_target():
    import torch
    from tilelang.tiletune import current_target

    if not torch.cuda.is_available():
        pytest.skip("CUDA or ROCm required")
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield current_target()
    torch.backends.cuda.matmul.allow_tf32 = previous


def _check(w, config, target, *, stress=False):
    import torch
    import tilelang
    from experiments.common.kernels import make_case

    case = make_case(w)
    inputs = case.inputs("cuda", torch.Generator(device="cuda").manual_seed(123))
    if stress:
        inputs[0].mul_(80)
    expected = case.reference(*inputs)
    kernel = tilelang.compile(
        case.build(**config), target=target, out_idx=case.out_idx, execution_backend="tvm_ffi", pass_configs=case.pass_configs
    )
    actual = kernel(*inputs)
    case.check(
        list(actual) if isinstance(actual, (tuple, list)) else [actual],
        list(expected) if isinstance(expected, (tuple, list)) else [expected],
    )
    return kernel


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_example_gemm_tail_inputs(gpu_target, dtype):
    w = Workload("gemm", "gemm", dict(m=97, n=113, k=81, transpose_b=True), dtype)
    _check(w, dict(block_M=64, block_N=64, block_K=32, thread_num=128, num_stages=1, enable_rasteration=False), gpu_target)


@pytest.mark.parametrize("causal,stages", [(False, 0), (False, 2), (True, 0), (True, 1)])
def test_example_attention_masks_and_tail_rows(gpu_target, causal, stages):
    w = Workload("attention", "attention", dict(batch=1, heads=2, sequence=137, dim=64, causal=causal))
    _check(w, dict(block_M=64, block_N=64, num_stages=stages, threads=128), gpu_target)


def test_example_fp8_gemm_native_schedule(gpu_target):
    if gpu_target["kind"] != "cuda" or int(gpu_target["arch"].split("_")[1].rstrip("af")) < 100:
        pytest.skip("TCGen05 requires Blackwell")
    w = Workload("fp8", "gemm_fp8", dict(m=256, n=256, k=256, transpose_b=True), dtype="float8_e4m3fn")
    _check(
        w,
        dict(
            block_M=128, block_N=256, block_K=128, num_stages=6, threads=128,
            implementation="tcgen05_2cta", group_size=1, use_tma_store=True, store_block_N=64
        ),
        gpu_target,
    )


def test_retired_rewrite_knobs_are_not_silently_ignored():
    from experiments.common.kernels import make_case

    cases = [
        (
            Workload("fp8", "gemm_fp8", dict(m=256, n=256, k=256, transpose_b=True), dtype="float8_e4m3fn"),
            dict(
                block_M=128, block_N=256, block_K=128, num_stages=6, threads=128,
                implementation="tcgen05_2cta", group_size=1, use_tma_store=True, store_block_N=64, vector=8
            ),
        ),
        (
            Workload("attention", "attention", dict(batch=1, heads=1, sequence=128, dim=64)),
            dict(block_M=64, block_N=64, num_stages=1, threads=128, qk_policy="square"),
        ),
        (
            Workload(
                "kda",
                "kda_chunk_intra_token_parallel",
                dict(batch=1, heads=1, sequence=128, dim=64, chunk_size=64, sub_chunk_size=16),
            ),
            dict(block_H=1, num_stages=0, threads=128, block_DK=64),
        ),
    ]
    for w, c in cases:
        with pytest.raises(TypeError, match="unexpected keyword"):
            make_case(w).build(**c)
