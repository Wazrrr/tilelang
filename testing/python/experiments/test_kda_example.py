"""KDA example output, masking and tail dimensions."""

import pytest
from experiments.common.spec import Workload


def _workload(op, dtype="float16", **parameters):
    p = dict(batch=2, heads=2, sequence=192, dim=64, value_dim=48, chunk_size=64)
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
    chunk = w.parameters["chunk_size"]
    a = inputs[3].reshape(w.parameters["batch"], -1, chunk, w.parameters["heads"], chunk)
    a.add_((torch.ones((chunk, chunk), device="cuda").triu(1) * 5).unsqueeze(1))
    expected = [case.reference(*inputs)]

    def execute(c):
        kernel = tilelang.compile(case.build(**c), target=target, out_idx=case.out_idx, execution_backend="tvm_ffi")
        return [kernel(*inputs)]

    actual = execute(config)
    case.check(actual, expected)
    if compare_baseline:
        baseline = execute(dict(block_DK=32, block_DV=32, num_stages=0, threads=128))
        case.check(baseline, expected)
        case.check(actual, baseline)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("block_k,block_v,stages,threads", [(32, 32, 0, 128), (64, 64, 2, 256), (32, 64, 3, 128)])
def test_chunk_example_matches_reference(gpu_target, dtype, block_k, block_v, stages, threads):
    config = dict(block_DK=block_k, block_DV=block_v, num_stages=stages, threads=threads)
    _run_and_check(_workload("kda_chunk_o", dtype), config, gpu_target)


@pytest.mark.parametrize("chunk,dim,value_dim", [(48, 96, 80), (64, 80, 65), (128, 160, 64)])
def test_chunk_example_tail_dimensions(gpu_target, chunk, dim, value_dim):
    w = _workload("kda_chunk_o", sequence=chunk * 3, chunk_size=chunk, dim=dim, value_dim=value_dim)
    config = dict(block_DK=32, block_DV=32, num_stages=0, threads=128)
    _run_and_check(w, config, gpu_target, compare_baseline=False)
