"""KDA schedule variants must preserve outputs, state, and chunk boundaries."""

from itertools import product

import pytest

from experiments.portable.spec import Device, TARGETS, Workload, configurations


def _workload(op, dtype="float16", **parameters):
    p = dict(batch=2, heads=2, sequence=67, dim=64, value_dim=48)
    if op == "kda_chunk_o":
        p.update(sequence=192, chunk_size=64)
    p.update(parameters)
    return Workload(op, op, p, dtype)


@pytest.mark.parametrize("op,baseline_count,total", [("kda_recurrent", 6, 114), ("kda_chunk_o", 24, 420)])
def test_expanded_grid_preserves_baseline_indices(op, baseline_count, total):
    grid = configurations(_workload(op), Device("ampere", TARGETS["ampere"]))
    if op == "kda_recurrent":
        baseline = [dict(block_v=v, threads=t) for v, t in product([16, 32, 64], [128, 256])]
    else:
        baseline = [dict(block_k=k, block_v=v, stages=s, threads=t) for k, v, s, t in product([32, 64], [32, 64], [0, 2, 3], [128, 256])]
    assert grid[:baseline_count] == baseline
    assert len(grid) == total
    assert len({tuple(sorted(c.items())) for c in grid}) == total
    assert all(c["implementation"] == "tiled" for c in grid[baseline_count:])
    if op == "kda_chunk_o":
        assert all(c["block_m"] * c["block_v"] >= 4 * c["threads"] for c in grid[baseline_count:])


@pytest.mark.parametrize("op", ["kda_recurrent", "kda_chunk_o"])
def test_tiled_candidates_through_portable_analysis(op, tmp_path):
    from dataclasses import replace

    from experiments.portable.run import make_request, run_native

    w, device = _workload(op), Device("ampere", TARGETS["ampere"])
    grid = configurations(w, device)
    w = replace(w, configs=[grid[6 if op == "kda_recurrent" else 24], grid[-1]])
    settings = dict(method="analyze", metric="traffic_waves", top_k=2, trace=False, memory_regime="streaming")
    result = run_native(make_request(w, device, settings), tmp_path)
    assert result["status"] == "analyzed", result
    assert result["configs"] == 2


@pytest.mark.parametrize(
    "op,config,match",
    [
        ("kda_recurrent", dict(block_t=3, unroll=2), "unroll must divide"),
        ("kda_recurrent", dict(block_t=0), "block_t must be"),
        ("kda_recurrent", dict(stages=-1), "stages must be"),
        ("kda_chunk_o", dict(block_m=17), "block_m must be"),
        ("kda_chunk_o", dict(block_s=0), "block_s must be"),
        ("kda_chunk_o", dict(intra_stages=-1), "intra_stages must be"),
        ("kda_recurrent", dict(implementation="typo"), "implementation must be"),
        ("kda_chunk_o", dict(implementation="baseline", block_m=16), "require implementation"),
    ],
)
def test_invalid_schedule_parameters(op, config, match):
    from experiments.portable.kernels import make_case

    kwargs = dict(block_v=32, threads=128, implementation="tiled")
    if op == "kda_chunk_o":
        kwargs.update(block_k=32, stages=0)
    with pytest.raises(ValueError, match=match):
        make_case(_workload(op)).build(**(kwargs | config))


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


def _run_and_check(w, config, target, *, compare_baseline=True, gate_mode="random"):
    import torch
    import tilelang
    from experiments.portable.kernels import make_case

    case = make_case(w)
    inputs = case.inputs("cuda", torch.Generator(device="cuda").manual_seed(123))
    if w.op == "kda_recurrent":
        if gate_mode == "identity":
            inputs[3].zero_()
            inputs[4].fill_(1)
        elif gate_mode == "no_update":
            inputs[4].zero_()
        elif gate_mode == "decay":
            inputs[3].fill_(-3)
    else:
        # Large upper-triangular entries expose a missing causal mask.
        chunk = w.parameters["chunk_size"]
        a = inputs[3].reshape(w.parameters["batch"], w.parameters["heads"], -1, chunk, chunk)
        a.add_(torch.ones((chunk, chunk), device="cuda").triu(1) * 5)
    reference = case.reference(*inputs)
    expected = list(reference) if isinstance(reference, tuple) else [reference]

    def execute(cfg):
        kernel = tilelang.compile(case.build(**cfg), target=target, out_idx=case.out_idx, execution_backend="tvm_ffi")
        result = kernel(*inputs)
        return list(result) if isinstance(result, (tuple, list)) else [result]

    actual = execute(dict(implementation="tiled", **config))
    case.check(actual, expected)
    if compare_baseline:
        baseline = dict(block_v=32, threads=128)
        if w.op == "kda_chunk_o":
            baseline.update(block_k=32, stages=0)
        old = execute(baseline)
        case.check(old, expected)
        case.check(actual, old)
    if w.op == "kda_recurrent":
        # The final state is FP32, so check it more tightly than FP16/BF16 O.
        torch.testing.assert_close(actual[1], expected[1], atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize(
    "block_v,threads,block_t,stages,unroll",
    [
        (16, 128, 4, 0, 1),
        (32, 256, 16, 2, 4),
        (64, 128, 32, 3, 1),
        (16, 256, 4, 3, 4),
    ],
)
def test_recurrent_matches_baseline_and_reference(gpu_target, dtype, block_v, threads, block_t, stages, unroll):
    config = dict(block_v=block_v, threads=threads, block_t=block_t, stages=stages, unroll=unroll)
    _run_and_check(_workload("kda_recurrent", dtype), config, gpu_target)


@pytest.mark.parametrize("sequence,dim,value_dim,gate_mode", [(1, 37, 19, "identity"), (19, 37, 19, "decay"), (17, 64, 33, "no_update")])
def test_recurrent_tail_and_gate_edges(gpu_target, sequence, dim, value_dim, gate_mode):
    w = _workload("kda_recurrent", sequence=sequence, dim=dim, value_dim=value_dim)
    config = dict(block_v=32, threads=128, block_t=16, stages=2, unroll=4)
    _run_and_check(w, config, gpu_target, compare_baseline=False, gate_mode=gate_mode)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    "block_m,block_k,block_v,block_s,stages,intra_stages,threads",
    [
        (16, 32, 32, 16, 0, 0, 128),
        (32, 64, 64, 32, 2, 2, 256),
        (64, 32, 32, 64, 3, 0, 128),
        (16, 64, 64, 64, 0, 2, 256),
        (64, 64, 32, 16, 3, 2, 256),
        (32, 32, 64, 16, 2, 0, 128),
    ],
)
def test_chunk_matches_baseline_and_reference(gpu_target, dtype, block_m, block_k, block_v, block_s, stages, intra_stages, threads):
    config = dict(
        block_m=block_m, block_k=block_k, block_v=block_v, block_s=block_s, stages=stages, intra_stages=intra_stages, threads=threads
    )
    _run_and_check(_workload("kda_chunk_o", dtype), config, gpu_target)


@pytest.mark.parametrize("chunk,dim,value_dim", [(19, 37, 23), (48, 80, 65), (8, 16, 8)])
def test_chunk_partial_tiles_do_not_cross_chunks(gpu_target, chunk, dim, value_dim):
    w = _workload("kda_chunk_o", sequence=chunk * 3, chunk_size=chunk, dim=dim, value_dim=value_dim)
    config = dict(block_m=32, block_k=32, block_v=32, block_s=32, stages=2, intra_stages=2, threads=128)
    _run_and_check(w, config, gpu_target, compare_baseline=False)


def test_chunk_long_pipelines(gpu_target):
    w = _workload("kda_chunk_o", sequence=256, chunk_size=128, dim=160, value_dim=64)
    config = dict(block_m=32, block_k=32, block_v=32, block_s=16, stages=3, intra_stages=3, threads=128)
    _run_and_check(w, config, gpu_target)
