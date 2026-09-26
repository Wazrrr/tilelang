"""Fixed-scale max fusion and lean scalar/matrix work capture."""

import copy
import json

import pytest
import tilelang.language as T

from tilelang.tiletune import analyze_prim_func, TileTuneConfig
from tiletune_core import rank_records, score_memory, score_work_max, select_top_k
from tiletune_core.work_max import WORK_KINDS
from test_analysis import gemm
from test_cost import LIMITS


ACCESS = [dict(operation=0, bytes=100, visits=2)]
RATES = dict(global_bytes_per_cycle=10, elementwise_ops_per_cycle=2)


def compute_facts(**amounts):
    return dict(work_per_cta={**dict.fromkeys(WORK_KINDS, 0), **amounts}, unknown=[], matrix_signatures=[], reduction_dtypes=[])


def score(amount=0, **kwargs):
    return score_work_max(ACCESS, compute_facts(elementwise_ops=amount), 9, 4, RATES, **kwargs)


def test_memory_and_compute_have_the_same_fixed_cycle_scale():
    memory = score(20)
    compute = score(80)
    assert memory["service_cycles"] == dict(memory=60, compute=30)
    assert memory["score"] == score_memory(ACCESS, 9, 4)["score"]
    assert compute["service_cycles"] == dict(memory=60, compute=120)
    assert compute["adjusted_logical_byte_waves"] == 1200
    assert compute["score"] > memory["score"]
    assert not compute["occupancy_gate_enabled"]


def test_normalization_is_pool_independent_and_common_scale_invariant():
    facts = compute_facts(elementwise_ops=80)
    original = copy.deepcopy(facts)
    base = score_work_max(ACCESS, facts, 9, 4, RATES)
    scaled = score_work_max(ACCESS, facts, 9, 4, {key: value * 7 for key, value in RATES.items()})
    assert base["score"] == scaled["score"]
    assert scaled["service_cycles"]["compute"] == pytest.approx(base["service_cycles"]["compute"] / 7)
    assert facts == original


def test_exact_rational_conversion_rounds_up_less_than_one_byte():
    result = score_work_max([], compute_facts(elementwise_ops=1), 1, 1, dict(global_bytes_per_cycle=10, elementwise_ops_per_cycle=3))
    assert result["compute_equivalent_byte_waves"] == 4
    assert result["max_service_cycles"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("kind", WORK_KINDS)
def test_nonzero_work_never_silently_uses_zero_or_another_rate(kind):
    result = score_work_max(ACCESS, compute_facts(**{kind: 10}), 1, 1, dict(global_bytes_per_cycle=10))
    assert result["score"] is None
    assert any(f"{kind}_per_cycle" in reason for reason in result["unknown"])


def test_zero_work_needs_no_compute_rates_but_unresolved_work_is_unknown():
    result = score_work_max(ACCESS, compute_facts(), 1, 1, dict(global_bytes_per_cycle=10))
    assert result["score"] is not None
    facts = compute_facts(elementwise_ops=None)
    assert score_work_max(ACCESS, facts, 1, 1, RATES)["score"] is None
    facts = compute_facts()
    facts["unknown"] = ["opaque computation"]
    assert score_work_max(ACCESS, facts, 1, 1, RATES)["score"] is None


@pytest.mark.parametrize("bad", [True, -1, float("nan"), float("inf")])
def test_invalid_rates_are_rejected(bad):
    with pytest.raises(ValueError):
        score_work_max(ACCESS, compute_facts(), 1, 1, dict(global_bytes_per_cycle=bad))


@pytest.mark.parametrize("field", ["a_dtype", "b_dtype", "accum_dtype", "instruction"])
def test_matrix_rate_requires_matching_signature(field):
    signature = dict(instruction="cuda.mma", a_dtype="bfloat16", b_dtype="bfloat16", accum_dtype="float32")
    facts = compute_facts(gemm_flops=100)
    facts["matrix_signatures"] = [signature]
    rates = dict(RATES, gemm_flops_per_cycle=50, gemm_signature=signature)
    assert score_work_max(ACCESS, facts, 1, 1, rates)["score"] is not None
    rates["gemm_signature"] = {**signature, field: "mismatch"}
    assert score_work_max(ACCESS, facts, 1, 1, rates)["score"] is None


def test_tcgen05_cannot_fall_back_to_mma_and_profiles_cannot_cross_targets():
    signature = dict(instruction="cuda.tcgen05", a_dtype="bfloat16", b_dtype="bfloat16", accum_dtype="float32")
    facts = compute_facts(tcgen05_gemm_flops=100)
    facts.update(matrix_signatures=[signature], target_arch="sm_103a")
    rates = dict(RATES, gemm_flops_per_cycle=50, gemm_signature={**signature, "instruction": "cuda.mma"}, profile_target="sm_103a")
    assert score_work_max(ACCESS, facts, 1, 1, rates)["score"] is None
    rates["tcgen05_gemm_flops_per_cycle"] = 100
    assert score_work_max(ACCESS, facts, 1, 1, rates)["score"] is not None
    rates["profile_target"] = "sm_90a"
    assert score_work_max(ACCESS, facts, 1, 1, rates)["score"] is None


def test_reduction_dtype_and_separate_max_rate_are_required():
    facts = compute_facts(reduction_max_ops=100)
    facts["reduction_dtypes"] = ["float32"]
    rates = dict(RATES, reduction_max_ops_per_cycle=10, reduction_dtype="float32")
    assert score_work_max(ACCESS, facts, 1, 1, rates)["score"] is not None
    facts["reduction_dtypes"] = ["float16"]
    assert score_work_max(ACCESS, facts, 1, 1, rates)["score"] is None


def test_pipeline_and_access_order_preserve_whole_boundary_ties():
    records = [
        dict(index=index, tile_cost=dict(score=score(amount, pipeline_depth=depth)["score"], ranking_metric="work_max"))
        for index, (amount, depth) in enumerate(((20, 3), (80, 3), (80, 3), (80, 2)))
    ]
    ranking = rank_records(records)
    assert [entry["rank"] for entry in ranking] == [1, 3, 3, 4]
    assert select_top_k(ranking, 2, strict_budget=True) == [0]
    assert select_top_k(ranking, 3, strict_budget=True) == [0, 1, 2]


def analyze(kernel, rates=None, **kwargs):
    settings = dict(ranking_metric="work_max", performance_model=rates or RATES, max_spill_bytes=None, max_local_bytes=None)
    settings.update(kwargs)
    return analyze_prim_func(kernel, settings, target=dict(kind="cuda", arch="sm_103a"), device_limits=LIMITS)


def test_scalar_parallel_and_serial_visits_count_once_without_addresses():
    @T.prim_func
    def kernel(source: T.Tensor((4096,), "float32"), output: T.Tensor((3, 8, 16), "float32")):
        with T.Kernel(1, threads=128):
            for step in T.serial(3):
                for row, column in T.Parallel(8, 16):
                    output[step, row, column] = T.exp2(source[step * 1024 + row * 32 + column * 2] * 2 + 1)

    result = analyze(kernel, dict(RATES, exp_ops_per_cycle=1))
    work = result["modules"]["compute_work"]
    assert not work["unknown"]
    assert work["work_per_cta"]["elementwise_ops"] == 3 * 8 * 16 * 2
    assert work["work_per_cta"]["exp_ops"] == 3 * 8 * 16
    assert result["tile_cost"]["score"] is not None


@pytest.mark.parametrize("clear", [True, False])
def test_logical_reduction_counts_include_accumulating_destination(clear):
    @T.prim_func
    def kernel(source: T.Tensor((8, 32), "float32"), output: T.Tensor((8,), "float32")):
        with T.Kernel(1, threads=128):
            local_source = T.alloc_fragment((8, 32), "float32")
            local_output = T.alloc_fragment((8,), "float32")
            T.copy(source, local_source)
            T.clear(local_output)
            for _step in T.serial(3):
                T.reduce_max(local_source, local_output, dim=1, clear=clear)
            T.copy(local_output, output)

    result = analyze(kernel, dict(RATES, reduction_max_ops_per_cycle=10))
    assert result["modules"]["compute_work"]["work_per_cta"]["reduction_max_ops"] == 3 * (248 if clear else 256)
    assert result["tile_cost"]["score"] is not None


def test_unknown_scalar_intrinsic_is_not_a_zero_compute_kernel():
    @T.prim_func
    def kernel(source: T.Tensor((128,), "float32"), output: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            for element in T.Parallel(128):
                output[element] = T.log(source[element])

    result = analyze(kernel)
    assert result["tile_cost"]["score"] is None
    assert any("unsupported scalar call" in reason for reason in result["tile_cost"]["unknown"])


def test_lean_analysis_and_portable_facts_do_not_require_a_gate_or_layout(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("work_max requested a heavyweight analysis")

    for name in (
        "pipeline.analyze_pipeline",
        "occupancy.analyze_waves",
        "memory.resident_warps_estimate",
        "memory.analyze_compute_intensity",
        "shared_memory.analyze_shared_memory",
        "engine.predict_warp_specialization",
        "engine.select_specialization",
        "ampere.prepare_ownership_analysis",
    ):
        monkeypatch.setattr(f"tilelang.tiletune.{name}", forbidden)
    signature = dict(instruction="cuda.mma", a_dtype="float16", b_dtype="float16", accum_dtype="float32")
    rates = dict(RATES, gemm_flops_per_cycle=100, tcgen05_gemm_flops_per_cycle=200, gemm_signature=signature)
    kernel = gemm(stages=2, extent=4)
    original = kernel.script()
    path = tmp_path / "work-max.json"
    result = analyze(kernel, rates, facts_path=str(path))
    facts = json.loads(path.read_text())
    replay = score_work_max(
        facts["accesses"], facts["compute"], facts["grid_blocks"], facts["sm_count"], facts["performance_model"], facts["pipeline_depth"]
    )
    assert facts["backend"] == "work_max.v1"
    assert result["tile_cost"]["score"] == replay["score"] is not None
    matrix_work = result["modules"]["compute_work"]["work_per_cta"]
    assert matrix_work["gemm_flops"] + matrix_work["tcgen05_gemm_flops"] == 2 * 32 * 32 * 32 * 4
    assert result["modules"]["pipeline_overlap"]["precision"] == "disabled"
    assert result["modules"]["waves"]["precision"] == "disabled"
    assert "bound" not in result["modules"]
    assert kernel.script() == original


def test_missing_profile_stays_unknown_and_default_remains_memory():
    result = analyze_prim_func(gemm(), dict(ranking_metric="work_max"), target=dict(kind="cuda", arch="sm_103a"), device_limits=LIMITS)
    assert result["tile_cost"]["score"] is None
    assert TileTuneConfig().ranking_metric == "memory"
    assert TileTuneConfig(ranking_metric="work_max").to_cache_key_dict() != TileTuneConfig().to_cache_key_dict()
    with pytest.raises(ValueError, match="family"):
        TileTuneConfig(ranking_metric="work_max", specialization="attention")


@pytest.mark.parametrize("architecture", ["sm_80", "sm_90a", "sm_100a", "sm_103", "sm_103a"])
def test_instruction_specific_profiles_without_device_execution(architecture):
    from test_pipeline import matrix_pipeline

    instruction = "cuda.wgmma" if architecture == "sm_90a" else "cuda.mma"
    rates = dict(
        RATES,
        gemm_flops_per_cycle=100,
        tcgen05_gemm_flops_per_cycle=200,
        gemm_signature=dict(instruction=instruction, a_dtype="float16", b_dtype="float16", accum_dtype="float32"),
        profile_target=architecture,
    )
    result = analyze_prim_func(
        matrix_pipeline(),
        dict(ranking_metric="work_max", performance_model=rates, max_spill_bytes=None, max_local_bytes=None),
        target=dict(kind="cuda", arch=architecture),
        device_limits=LIMITS,
    )
    assert result["tile_cost"]["score"] is not None
    assert not result["modules"]["ranking"]["occupancy_gate_enabled"]


def test_subwave_launch_does_not_add_a_hidden_occupancy_gate():
    facts = compute_facts(elementwise_ops=80)
    single = score_work_max(ACCESS, facts, 1, 8, RATES)
    full = score_work_max(ACCESS, facts, 8, 8, RATES)
    assert single["score"] == full["score"]
    assert single["occupancy_penalty"] == full["occupancy_penalty"] == 1
