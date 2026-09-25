"""KDA intra example coefficients, masking and tail dimensions."""

import pytest
from experiments.common.spec import Workload


def _workload(op, dtype="float16", **parameters):
    p = dict(batch=2, heads=2, sequence=192, dim=64, chunk_size=64, sub_chunk_size=16)
    p.update(parameters)
    return Workload(op, op, p, dtype)


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


def _run_and_check(w, config, target, *, compare_baseline=True):
    import torch
    import tilelang
    from experiments.common.kernels import make_case

    case = make_case(w)
    inputs = case.inputs("cuda", torch.Generator(device="cuda").manual_seed(123))
    expected = [case.reference(*inputs)]

    def execute(c):
        kernel = tilelang.compile(
            case.build(**c),
            target=target,
            out_idx=case.out_idx,
            execution_backend="tvm_ffi",
            pass_configs=case.pass_configs,
        )
        return [kernel(*inputs)]

    actual = execute(config)
    case.check(actual, expected)
    if compare_baseline:
        baseline = execute(dict(block_H=1, num_stages=0, threads=128))
        case.check(baseline, expected)
        case.check(actual, baseline)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("block_h,stages,threads", [(1, 0, 128), (2, 2, 256), (4, 3, 128)])
def test_intra_example_matches_reference(gpu_target, dtype, block_h, stages, threads):
    config = dict(block_H=block_h, num_stages=stages, threads=threads)
    _run_and_check(_workload("kda_chunk_intra_token_parallel", dtype), config, gpu_target)


@pytest.mark.parametrize("chunk,sub_chunk,heads,block_h", [(48, 16, 6, 3), (64, 8, 4, 2), (128, 32, 5, 5)])
def test_intra_example_tail_dimensions(gpu_target, chunk, sub_chunk, heads, block_h):
    w = _workload(
        "kda_chunk_intra_token_parallel",
        sequence=chunk * 3,
        chunk_size=chunk,
        sub_chunk_size=sub_chunk,
        dim=128,
        heads=heads,
    )
    config = dict(block_H=block_h, num_stages=0, threads=128)
    _run_and_check(w, config, gpu_target, compare_baseline=False)
