"""Fixed-work and underfill rank fusion without changing either component."""

import copy
import json

import pytest

from tilelang.tiletune import TileTuneConfig, analyze_prim_func
from tilelang.tiletune.runtime import TileTuneSession
from tiletune_core import rank_records, score_memory, score_work_max, score_work_rank_product, select_top_k
from test_analysis import gemm
from test_cost import LIMITS
from test_work_max import ACCESS, RATES, compute_facts


def records_for(pairs):
    return [
        dict(
            index=index,
            status="analyzed",
            tile_cost=dict(
                ranking_metric="work_rank_product",
                score=None,
                component_scores=dict(work_max=work, underfill=underfill),
            ),
        )
        for index, (work, underfill) in enumerate(pairs)
    ]


def test_components_match_unchanged_work_max_and_underfill_scores():
    facts = compute_facts(elementwise_ops=80)
    original = copy.deepcopy(facts)
    result = score_work_rank_product(iter(ACCESS), facts, 9, 4, RATES, 2)
    assert result["component_scores"] == dict(
        work_max=score_work_max(ACCESS, facts, 9, 4, RATES, 2)["score"],
        underfill=score_memory(ACCESS, 9, 4, 2, launch_underfill=True)["score"],
    )
    assert result["score"] is None and result["score_scope"] == "candidate_pool"
    assert not result["occupancy_gate_enabled"]
    assert facts == original


def test_fusion_uses_tail_ranks_and_preserves_input_records():
    records = records_for([(10, 40), (10, 20), (30, 20), (40, 10)])
    original = copy.deepcopy(records)
    ranking = rank_records(records)
    assert [entry["index"] for entry in ranking] == [3, 1, 0, 2]
    assert [entry["score"] for entry in ranking] == [4, 6, 8, 9]
    assert ranking[1]["component_tail_ranks"] == dict(work_max=2, underfill=3)
    assert all(entry["score_scope"] == "candidate_pool" for entry in ranking)
    assert records == original


def test_fused_product_ties_are_not_split_at_the_budget():
    ranking = rank_records(records_for([(1, 1), (2, 5), (3, 3), (4, 4), (5, 2)]))
    assert [entry["rank"] for entry in ranking] == [1, 2, 4, 4, 5]
    assert select_top_k(ranking, 3, include_unknown=False, strict_budget=True) == [0, 2]


def test_fusion_ignores_latency_resources_and_component_scale():
    records = records_for([(10, 40), (10, 20), (30, 20), (40, 10)])
    expected = rank_records(records)
    for record in records:
        record.update(latency_ms=record["index"] + 1, compiler_resources={"registers": 255})
        components = record["tile_cost"]["component_scores"]
        components["work_max"] = 3 * components["work_max"] + 20
        components["underfill"] = 7 * components["underfill"] + 90
    assert rank_records(list(reversed(records))) == expected


@pytest.mark.parametrize("component", ["work_max", "underfill"])
@pytest.mark.parametrize("value", [None, True, -1, float("nan"), float("inf")])
def test_incomplete_components_are_excluded_from_both_views(component, value):
    records = records_for([(10, 20), (20, 10), (0, 0)])
    records[2]["tile_cost"]["component_scores"][component] = value
    ranking = rank_records(records)
    assert [entry["score"] for entry in ranking] == [2, 2, None]
    assert ranking[-1]["tier"] == "unknown"


def test_missing_compute_rate_never_falls_back_to_memory_only():
    result = score_work_rank_product(ACCESS, compute_facts(elementwise_ops=80), 9, 4, dict(global_bytes_per_cycle=10))
    assert result["component_scores"]["work_max"] is None
    assert result["component_scores"]["underfill"] is not None
    ranking = rank_records([dict(index=0, tile_cost={**result, "ranking_metric": "work_rank_product"})])
    assert ranking[0]["tier"] == "unknown" and ranking[0]["score"] is None


def test_failures_and_rejections_do_not_shift_component_ranks():
    records = records_for([(10, 20), (20, 10), (0, 0), (0, 0)])
    records[2]["pre_lowering"] = dict(would_reject=True)
    records[3]["status"] = "analysis_failed"
    ranking = rank_records(records)
    assert [entry["score"] for entry in ranking] == [2, 2, None, None]
    assert [entry["tier"] for entry in ranking] == ["eligible", "eligible", "pressure_rejected", "unavailable"]


def test_mixed_fusion_units_are_rejected():
    records = records_for([(1, 2), (2, 1)])
    records[1]["tile_cost"]["ranking_metric"] = "rank_product"
    with pytest.raises(ValueError, match="different ranking metrics"):
        rank_records(records)


def test_lean_analysis_exports_replayable_facts_without_extra_models(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("work rank product requested a gate or heavyweight analysis")

    for name in (
        "pipeline.analyze_pipeline", "occupancy.analyze_waves", "memory.resident_warps_estimate",
        "memory.analyze_compute_intensity", "shared_memory.analyze_shared_memory",
        "engine.predict_warp_specialization", "engine.select_specialization", "ampere.prepare_ownership_analysis",
    ):
        monkeypatch.setattr(f"tilelang.tiletune.{name}", forbidden)
    signature = dict(instruction="cuda.mma", a_dtype="float16", b_dtype="float16", accum_dtype="float32")
    rates = dict(RATES, gemm_flops_per_cycle=100, tcgen05_gemm_flops_per_cycle=200, gemm_signature=signature)
    kernel = gemm(stages=2, extent=4)
    original = kernel.script()
    path = tmp_path / "work-rank-product.json"
    result = analyze_prim_func(
        kernel,
        dict(ranking_metric="work_rank_product", performance_model=rates, facts_path=str(path), max_spill_bytes=None, max_local_bytes=None),
        target=dict(kind="cuda", arch="sm_103a"), device_limits=LIMITS,
    )
    facts = json.loads(path.read_text())
    expected = score_work_rank_product(
        facts["accesses"], facts["compute"], facts["grid_blocks"], facts["sm_count"], facts["performance_model"], facts["pipeline_depth"]
    )
    assert facts["backend"] == "work_rank_product.v1"
    assert result["tile_cost"]["component_scores"] == expected["component_scores"]
    assert all(score is not None for score in expected["component_scores"].values())
    assert result["tile_cost"]["score"] is None
    assert result["tile_cost"]["score_scope"] == "candidate_pool"
    assert "compute_work" in result["modules"] and "bound" not in result["modules"]
    assert result["modules"]["waves"]["precision"] == "disabled"
    assert result["modules"]["pipeline_overlap"]["precision"] == "disabled"
    assert kernel.script() == original


def test_runtime_uses_original_half_pool_without_kernel_execution(monkeypatch):
    configs = [dict(identifier=index) for index in range(6)]
    session = TileTuneSession(TileTuneConfig(ranking_metric="work_rank_product", alpha=0.5), configs)
    records = records_for([(1, 1), (2, 6), (3, 2), (4, 5), (5, 3), (6, 4)])

    def elaborate(index, kwargs, elaborate_func, **unused):
        session.records[index].update(records[index])
        return index

    monkeypatch.setattr(session, "elaborate", elaborate)
    selected = session.prepare_top_k([(index, config, {}) for index, config in enumerate(configs)], lambda identifier: identifier)
    assert selected == [0, 2, 1]
    assert session.selection["requested_k"] == 3
    assert session.selection["strict_budget"]


def test_new_metric_is_opt_in_with_separate_cache_identity():
    assert TileTuneConfig().ranking_metric == "memory"
    keys = [TileTuneConfig(ranking_metric=metric).to_cache_key_dict() for metric in ("memory", "rank_product", "work_max", "work_rank_product")]
    assert len({json.dumps(key, sort_keys=True) for key in keys}) == 4
    with pytest.raises(ValueError, match="family"):
        TileTuneConfig(ranking_metric="work_rank_product", specialization="attention")
