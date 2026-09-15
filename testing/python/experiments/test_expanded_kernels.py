"""New schedule parameters must preserve tails, masking and reductions on GPU."""

import pytest

from experiments.portable.spec import Workload


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
    from experiments.portable.kernels import make_case

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


@pytest.mark.parametrize(
    "policy,swizzle,stages,ta,tb,dtype",
    [
        ("square", 4, 1, False, False, "float16"),
        ("full_row", 8, 4, True, False, "float16"),
        ("full_col", 0, 2, False, True, "bfloat16"),
    ],
)
def test_gemm_new_policies_and_epilogue_tails(gpu_target, policy, swizzle, stages, ta, tb, dtype):
    w = Workload("gemm", "gemm", dict(m=97, n=113, k=81, batch=2, transpose_a=ta, transpose_b=tb, epilogue="bias_relu"), dtype)
    _check(w, dict(block_m=64, block_n=64, block_k=16, threads=128, stages=stages, warp_policy=policy, swizzle_panel=swizzle), gpu_target)


@pytest.mark.parametrize("qk", ["square", "full_row", "full_col"])
@pytest.mark.parametrize("pv", ["square", "full_row", "full_col"])
def test_attention_independent_partitions_with_causal_tails(gpu_target, qk, pv):
    w = Workload("attention", "attention", dict(batch=1, heads=2, sequence=137, dim=64, causal=True))
    _check(w, dict(block_M=32, block_N=64, num_stages=2, threads=128, qk_policy=qk, pv_policy=pv, copy_width=2), gpu_target)


@pytest.mark.parametrize(
    "m,n,threads,stages,copy_width,causal,dtype,dim",
    [
        (16, 256, 256, 0, None, True, "float16", 64),
        (32, 128, 256, 1, 4, True, "float16", 64),
        (32, 16, 128, 3, None, True, "float16", 64),
        (32, 16, 128, 0, None, True, "float16", 64),
        (64, 32, 256, 1, 8, False, "bfloat16", 64),
        (32, 64, 128, 2, 2, True, "float16", 128),
        (32, 32, 128, 3, 4, False, "bfloat16", 32),
    ],
)
def test_attention_softmax_layout_handles_wide_and_narrow_reduction_segments(
    gpu_target, m, n, threads, stages, copy_width, causal, dtype, dim
):
    w = Workload("attention", "attention", dict(batch=1, heads=2, sequence=137, dim=dim, causal=causal), dtype)
    _check(
        w,
        dict(
            implementation="tiled",
            block_M=m,
            block_N=n,
            num_stages=stages,
            threads=threads,
            qk_policy="full_col",
            pv_policy="full_row",
            copy_width=copy_width,
        ),
        gpu_target,
    )


@pytest.mark.parametrize("op", ["softmax", "rmsnorm", "reduce_sum", "elementwise"])
@pytest.mark.parametrize("row_threads,vector,dtype", [(1, 1, "float16"), (2, 4, "bfloat16"), (4, 8, "float32")])
def test_column_tiles_and_thread_ownership_preserve_full_row_semantics(gpu_target, op, row_threads, vector, dtype):
    w = Workload(op, op, dict(rows=5, columns=2053), dtype)
    _check(
        w,
        dict(
            implementation="tiled" if op == "elementwise" else "streamed",
            block_rows=4,
            block_cols=512,
            threads=128,
            vector=vector,
            row_threads=row_threads,
        ),
        gpu_target,
        stress=op == "softmax",
    )


@pytest.mark.parametrize(
    "op,config",
    [
        ("kda_recurrent", dict(implementation="tiled", block_v=8, block_t=2, unroll=2, stages=1, threads=64)),
        ("kda_recurrent", dict(implementation="tiled", block_v=128, block_t=8, unroll=8, stages=4, threads=128)),
        (
            "kda_chunk_o",
            dict(implementation="tiled", block_m=128, block_k=16, block_v=128, block_s=16, stages=1, intra_stages=1, threads=128),
        ),
    ],
)
def test_extended_kda_axes_preserve_state_and_chunk_tails(gpu_target, op, config):
    p = dict(batch=1, heads=2, sequence=19, dim=37, value_dim=23)
    if op == "kda_chunk_o":
        p.update(sequence=96, chunk_size=48)
    _check(Workload(op, op, p), config, gpu_target)


@pytest.mark.parametrize("intra_stages", [0, 1, 2, 3])
@pytest.mark.parametrize("block_m,block_s,block_v", [(16, 64, 128), (32, 128, 32)])
def test_kda_short_causal_prefixes_keep_pipeline_indices_aligned(gpu_target, intra_stages, block_m, block_s, block_v):
    w = Workload("kda_chunk", "kda_chunk_o", dict(batch=1, heads=2, sequence=256, dim=128, value_dim=128, chunk_size=128))
    _check(
        w,
        dict(
            implementation="tiled",
            block_m=block_m,
            block_k=32,
            block_v=block_v,
            block_s=block_s,
            stages=0,
            intra_stages=intra_stages,
            threads=128,
        ),
        gpu_target,
    )


def test_row_knobs_cannot_silently_do_nothing():
    from experiments.portable.kernels import make_case

    case = make_case(Workload("softmax", "softmax", dict(rows=4, columns=128)))
    with pytest.raises(ValueError, match="require a non-baseline"):
        case.build(block_rows=1, threads=128, block_cols=128)
    with pytest.raises(ValueError, match="cannot supply"):
        case.build(block_rows=1, threads=128, implementation="streamed", block_cols=128, vector=8)
