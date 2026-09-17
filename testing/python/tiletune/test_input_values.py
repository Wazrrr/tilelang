"""Declared metadata resolves addresses without changing the executable kernel."""

import pytest
import tilelang.language as T
from tilelang.tiletune import analyze_prim_func, TileTuneConfig
from experiments.common.spec import Workload
from experiments.grouped_gemm.kernel import make_case
from test_cost import LIMITS
from test_modules import PROFILE, TARGET
from experiments.common.spec import default_workloads


@pytest.mark.parametrize("workload", default_workloads(), ids=lambda w: w.name)
def test_every_final_workload_has_a_scored_native_configuration(workload):
    from experiments.common.kernels import make_case as make_workload

    configs = {
        "gemm": dict(block_M=128, block_N=128, block_K=64, thread_num=128, num_stages=0, enable_rasteration=False),
        "gemm_fp8": dict(block_M=128, block_N=128, block_K=64, threads=128, num_stages=0, enable_rasteration=False),
        "attention": dict(block_M=64, block_N=64, threads=128, num_stages=0),
        "kda_chunk_o": dict(block_DK=64, block_DV=64, threads=128, num_stages=0),
        "grouped_gemm": dict(block_M=64, block_N=128, block_K=64, threads=128, num_stages=0),
    }
    rates = dict(
        PROFILE,
        gemm_rates=[
            dict(
                signature=dict(instruction=instruction, a_dtype=dtype, b_dtype=dtype, accum_dtype="float32"),
                rates=dict(gemm_flops_per_cycle=2048),
            )
            for instruction in ("cuda.mma", "cuda.wgmma")
            for dtype in ("float16", "float8_e4m3fn")
        ],
    )
    case = make_workload(workload)
    result = analyze_prim_func(
        case.build(**configs[workload.op]),
        dict(performance_model=rates, input_values=case.input_values or None),
        target=TARGET,
        device_limits=LIMITS,
    )
    assert result["tile_cost"]["score"] is not None, result["diagnostics"]
    assert not result["pressure"]["decision"]["would_reject"]


@pytest.mark.parametrize("sizes,transpose_b", [([64, 128], False), ([63, 77, 111, 280], True)])
def test_grouped_work_counts_executed_padding_and_exact_output(sizes, transpose_b):
    # Enumerate the example's global addresses independently of the analyzer.
    n, k, bm, bn, bk = 128, 96, 64, 64, 32
    w = Workload("metadata", "grouped_gemm", dict(batch_sizes=sizes, n=n, k=k, transpose_b=transpose_b))
    case = make_case(w)
    func = case.build(block_M=bm, block_N=bn, block_K=bk, threads=128, num_stages=0)
    before = func.script()
    result = analyze_prim_func(func, dict(performance_model=PROFILE, input_values=case.input_values), target=TARGET, device_limits=LIMITS)
    assert func.script() == before
    assert result["tile_cost"]["score"] is not None, result["diagnostics"]
    p = result["modules"]["pipeline_overlap"]
    totals = {
        key: sum(p["region_work"][g["iterations"]][key] * g["count"] for g in p["cta_work"]["groups"]) * p["cta_work"]["repetitions"]
        for key in ("read_bytes", "write_bytes", "gemm_flops")
    }
    rows, tiles, start = 0, 0, 0
    for size in sizes:
        for offset in range(0, size, bm):
            rows += sum(start + offset + row < sum(sizes) for row in range(bm))
            tiles += 1
        start += size
    blocks = tiles * (n // bn)
    # The example loads full A tiles, even across group boundaries. Only the
    # end of the packed allocation is masked; output stores mask every group.
    metadata_bytes = (len(sizes) + 4) * 4 * blocks
    assert totals["read_bytes"] == rows * k * 2 * (n // bn) + blocks * bn * k * 2 + metadata_bytes
    assert totals["write_bytes"] == sum(sizes) * n * 2
    assert totals["gemm_flops"] == blocks * 2 * bm * bn * k


def metadata_copy():
    @T.prim_func
    def kernel(X: T.Tensor((8,), "float32"), Sizes: T.Tensor((2,), "int32"), Y: T.Tensor((8,), "float32")):
        with T.Kernel(2, threads=128) as bx:
            for i in T.Parallel(4):
                if i < Sizes[bx]:
                    Y[bx * 4 + i] = X[bx * 4 + i]

    return kernel


def test_unknown_metadata_is_not_guessed_and_declared_values_change_identity():
    func = metadata_copy()
    result = analyze_prim_func(func, dict(performance_model=PROFILE), target=TARGET, device_limits=LIMITS)
    assert result["tile_cost"]["score"] is None
    a, b = TileTuneConfig(input_values={"1": [1, 4]}), TileTuneConfig(input_values={"1": [4, 4]})
    assert a.to_cache_key_dict() != b.to_cache_key_dict()


@pytest.mark.parametrize("values", [{"-1": [1]}, {"x": [1]}, {"1": []}, {"1": [True]}, {"1": [1.5]}])
def test_metadata_contract_rejects_invalid_values(values):
    with pytest.raises(ValueError, match="input_values"):
        TileTuneConfig(input_values=values)


@pytest.mark.parametrize("values", [{"0": [1] * 8}, {"1": [1]}, {"9": [1]}])
def test_metadata_contract_checks_parameter_dtype_shape_and_index(values):
    with pytest.raises(ValueError, match="input_values"):
        analyze_prim_func(metadata_copy(), dict(input_values=values), target=TARGET)


def test_metadata_cannot_be_a_writable_parameter():
    @T.prim_func
    def kernel(Sizes: T.Tensor((2,), "int32")):
        with T.Kernel(1, threads=128):
            Sizes[0] = 1

    with pytest.raises(ValueError, match="read-only"):
        analyze_prim_func(kernel, dict(input_values={"0": [1, 4]}), target=TARGET)


def test_experiment_verifies_actual_metadata_before_execution():
    import torch

    w = Workload("metadata", "grouped_gemm", dict(batch_sizes=[1, 63, 65], n=32, k=32))
    case = make_case(w)
    inputs = case.inputs("cpu", torch.Generator().manual_seed(123))
    case.check_input_values(inputs)
    inputs[4][1] += 1
    with pytest.raises(ValueError, match="declared TileTune metadata"):
        case.check_input_values(inputs)


def test_kda_tails_count_both_products_padded_compute_and_masked_traffic():
    from experiments.kda.cases import cases
    from experiments.kda.kernel import make_case as make_kda

    case = make_kda(cases(holdout=True)[1])
    func = case.build(block_DK=64, block_DV=64, num_stages=0, threads=128)
    result = analyze_prim_func(func, dict(performance_model=PROFILE), target=TARGET, device_limits=LIMITS)
    assert result["tile_cost"]["score"] is not None, result["diagnostics"]
    p = result["modules"]["pipeline_overlap"]
    totals = {
        key: sum(p["region_work"][g["iterations"]][key] * g["count"] for g in p["cta_work"]["groups"]) * p["cta_work"]["repetitions"]
        for key in ("read_bytes", "write_bytes", "gemm_flops", "exp_ops")
    }
    chunks, heads, s, k, v = 16, 4, 48, 96, 80
    blocks = chunks * heads * 2
    assert totals["read_bytes"] == blocks * (s * k * 6 + s * s * 2) + chunks * heads * (k * v * 2 + s * v * 2)
    assert totals["write_bytes"] == chunks * heads * s * v * 2
    assert totals["gemm_flops"] == blocks * 2 * s * 64 * (128 + s)
    assert totals["exp_ops"] == blocks * s * 128
