"""KDA intra-chunk outputs, sub-chunk masks, head tails and input contracts."""

import pytest
from experiments.common.spec import Workload

OP = "kda_chunk_intra_token_parallel"


def _workload(dtype="bfloat16", **parameters):
    p = dict(batch=2, heads=5, sequence=192, dim=128, chunk_size=64, sub_chunk_size=16)
    p.update(parameters)
    return Workload("kda_intra", OP, p, dtype)


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


def _check_masks(outputs, w):
    import torch

    aqk, akk = outputs
    p = w.parameters
    chunk, sub = p["chunk_size"], p["sub_chunk_size"]
    tokens = torch.arange(p["sequence"], device=aqk.device)
    aqk_cols = torch.arange(chunk, device=aqk.device)
    aqk_mask = (aqk_cols[None, :] // sub == (tokens % chunk)[:, None] // sub) & (aqk_cols[None, :] <= (tokens % chunk)[:, None])
    akk_mask = torch.arange(sub, device=akk.device)[None, :] < (tokens % sub)[:, None]
    assert torch.count_nonzero(aqk.masked_select(~aqk_mask[None, :, None, :])) == 0
    assert torch.count_nonzero(akk.masked_select(~akk_mask[None, :, None, :])) == 0


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("block_h,stages,threads", [(1, 0, 32), (4, 1, 128), (8, 3, 256), (3, 1, 128)])
def test_intra_example_matches_reference(gpu_target, dtype, block_h, stages, threads):
    import torch
    import tilelang
    from experiments.common.kernels import make_case

    w = _workload(dtype)
    case = make_case(w)
    inputs = case.inputs("cuda", torch.Generator(device="cuda").manual_seed(123))
    kernel = tilelang.compile(
        case.build(block_H=block_h, num_stages=stages, threads=threads),
        target=gpu_target,
        out_idx=case.out_idx,
        execution_backend="tvm_ffi",
        pass_configs=case.pass_configs,
    )
    actual, expected = kernel(*inputs), case.reference(*inputs)
    assert len(actual) == 2
    assert actual[0].shape == (2, 192, 5, 64)
    assert actual[1].shape == (2, 192, 5, 16)
    case.check(actual, expected)
    _check_masks(actual, w)


def test_reference_matches_direct_token_pair_formula():
    import torch
    from experiments.common.kernels import make_case

    w = _workload(batch=1, heads=2, sequence=16, dim=8, chunk_size=8, sub_chunk_size=4)
    case = make_case(w)
    q, k, gates, beta = case.inputs("cpu", torch.Generator().manual_seed(123))
    aqk, akk = case.reference(q, k, gates, beta)
    expected_qk, expected_kk = torch.zeros_like(aqk), torch.zeros_like(akk)
    for i in range(16):
        for j in range(i // 4 * 4, i + 1):
            gated_key = k[0, j].float() * torch.exp2(gates[0, i] - gates[0, j])
            expected_qk[0, i, :, j % 8] = (q[0, i].float() * gated_key).sum(-1) * 8**-0.5
            if j < i:
                expected_kk[0, i, :, j % 4] = (k[0, i].float() * gated_key).sum(-1) * beta[0, i].float()
    case.check([aqk, akk], [expected_qk, expected_kk])
    _check_masks([aqk, akk], w)


def test_intra_group_of_eight_checks_both_outputs_and_compiler_resources(gpu_target):
    import torch
    from tvm.target import Target
    from tilelang.autotuner.grouped_compile import compile_grouped_unit_tvm_ffi
    from tilelang.autotuner.param import CompileArgs
    from tilelang.tiletune import TileTuneConfig, query_device_limits
    from tilelang.tiletune.runtime import TileTuneSession
    from experiments.common.kernels import make_case
    from experiments.common.resource_policy import h200_post_compile_policy

    if gpu_target["kind"] != "cuda":
        pytest.skip("grouped compilation requires CUDA")
    workload = _workload()
    case = make_case(workload)
    configs = [dict(block_H=h, num_stages=1, threads=t) for h in (1, 2, 4, 8) for t in (128, 256)]
    config = TileTuneConfig(
        enabled=True,
        mode="report_only",
        ranking_metric="memory",
        max_spill_bytes=None,
        max_local_bytes=None,
        post_compile_policy=h200_post_compile_policy(workload, gpu_target),
    )
    session = TileTuneSession(config, configs, target=gpu_target, device_limits=query_device_limits(gpu_target))
    results = compile_grouped_unit_tvm_ffi(
        list(enumerate(configs)),
        CompileArgs(target=Target(gpu_target), out_idx=case.out_idx, pass_configs=case.pass_configs, execution_backend="tvm_ffi"),
        case.build,
        tiletune_session=session,
    )
    assert len(results) == 8
    inputs = case.inputs("cuda", torch.Generator(device="cuda").manual_seed(123))
    expected = case.reference(*inputs)
    for idx, _, kernel, error in results:
        assert error is None, str(error)
        assert kernel.adapter._autotune_group_size == 8
        assert session.records[idx]["tile_cost"]["score"] is not None
        assert session.records[idx]["post_compile"]["status"] == "pass"
        case.check(kernel(*inputs), expected)


@pytest.mark.parametrize(
    "parameters,message",
    [
        (dict(sequence=65), "complete chunks"),
        (dict(sub_chunk_size=15), "complete sub-chunks"),
    ],
)
def test_intra_requires_complete_chunks_and_sub_chunks(parameters, message):
    with pytest.raises(ValueError, match=message):
        _workload(**parameters)


def test_intra_carver_does_not_use_chunk_output_model(tmp_path):
    from experiments.common.baselines import carver_support_reason
    from experiments.common.run import make_request, run_native
    from experiments.common.spec import Device, TARGETS

    workload, device = _workload(), Device("hopper", TARGETS["hopper"])
    assert "intra-chunk" in carver_support_reason(workload, device)
    result = run_native(make_request(workload, device, dict(method="carver")), tmp_path)
    assert result["status"] == "unsupported"
    assert "chunk output" in result["reason"]
    assert not list(tmp_path.iterdir())
