"""Pool-scoped rank fusion must not change budgets, policies, or kernel facts."""

import copy
import json

import pytest

from tilelang.tiletune import analyze_prim_func, TileTuneConfig
from tilelang.tiletune.runtime import TileTuneSession
from tiletune_core import rank_records, score_memory, score_rank_product, select_top_k
from test_analysis import gemm
from test_cost import LIMITS


def records_for(pairs):
    return [
        dict(
            index=index,
            status="analyzed",
            tile_cost=dict(
                ranking_metric="rank_product",
                score=None,
                component_scores=dict(memory=memory, underfill=underfill),
            ),
        )
        for index, (memory, underfill) in enumerate(pairs)
    ]


def test_fusion_uses_component_tail_ranks_not_dense_ranks():
    records = records_for([(10, 40), (10, 20), (30, 20), (40, 10)])
    original = copy.deepcopy(records)
    ranking = rank_records(records)
    assert [entry["index"] for entry in ranking] == [3, 1, 0, 2]
    assert [entry["score"] for entry in ranking] == [4, 6, 8, 9]
    assert ranking[1]["component_tail_ranks"] == dict(memory=2, underfill=3)
    assert all(entry["score_scope"] == "candidate_pool" for entry in ranking)
    assert records == original


def test_product_collisions_remain_whole_boundary_groups():
    ranking = rank_records(records_for([(1, 1), (2, 5), (3, 3), (4, 4), (5, 2)]))
    assert [entry["score"] for entry in ranking] == [1, 9, 10, 10, 16]
    assert [entry["rank"] for entry in ranking] == [1, 2, 4, 4, 5]
    assert select_top_k(ranking, 3, strict_budget=True) == [0, 2]
    assert select_top_k(ranking, 4, strict_budget=True) == [0, 2, 1, 4]


def test_fusion_is_scale_order_and_measurement_independent():
    records = records_for([(10, 40), (10, 20), (30, 20), (40, 10)])
    expected = rank_records(records)
    for record in records:
        record.update(latency_ms=100 - record["index"], compiler_resources={"registers": 255})
        record["tile_cost"]["score"] = 1000 - record["index"]
        components = record["tile_cost"]["component_scores"]
        components["memory"] = 17 * components["memory"] + 2
        components["underfill"] = 13 * components["underfill"] + 101
    assert rank_records(list(reversed(records))) == expected
    subset = {entry["index"]: entry for entry in rank_records(records[1:])}
    assert subset[1]["score"] == 3
    assert subset[1]["score"] != next(entry["score"] for entry in expected if entry["index"] == 1)


@pytest.mark.parametrize(
    "components",
    [
        None,
        {},
        [],
        {"memory": 1},
        {"memory": None, "underfill": 1},
        {"memory": float("nan"), "underfill": 1},
        {"memory": 1, "underfill": float("inf")},
        {"memory": True, "underfill": 1},
        {"memory": -1, "underfill": 1},
        {"memory": 1, "underfill": "2"},
    ],
)
def test_incomplete_components_are_excluded_from_both_views(components):
    records = records_for([(10, 20), (20, 10), (0, 0)])
    records[2]["tile_cost"]["component_scores"] = components
    ranking = rank_records(records)
    assert [entry["score"] for entry in ranking] == [2, 2, None]
    assert ranking[-1]["tier"] == "unknown"
    assert select_top_k(ranking, 3, include_unknown=False, strict_budget=True) == [0, 1]


def test_pressure_rejections_and_failures_do_not_shift_component_ranks():
    records = records_for([(10, 20), (20, 10), (0, 0), (0, 0)])
    records[2]["pre_lowering"] = dict(would_reject=True)
    records[3]["status"] = "analysis_failed"
    ranking = rank_records(records)
    assert [entry["tier"] for entry in ranking] == ["eligible", "eligible", "pressure_rejected", "unavailable"]
    assert [entry["score"] for entry in ranking] == [2, 2, None, None]
    assert select_top_k(ranking, 4, include_unknown=False, strict_budget=True) == [0, 1]


def test_fusion_rejects_mixed_metrics_and_duplicate_indices():
    records = records_for([(1, 2)])
    with pytest.raises(ValueError, match="different ranking metrics"):
        rank_records(records + [dict(index=1, tile_cost=dict(score=10))])
    with pytest.raises(ValueError, match="unique original indices"):
        rank_records(records * 2)


def test_components_support_iterables_and_resolved_zero_work():
    accesses = [dict(operation=0, bytes=1024, visits=3)]
    result = score_rank_product(iter(accesses), 128, 148, 2)
    assert result["component_scores"] == dict(
        memory=score_memory(accesses, 128, 148, 2)["score"],
        underfill=score_memory(accesses, 128, 148, 2, launch_underfill=True)["score"],
    )
    assert all(value is not None for value in score_rank_product([], 1, 148)["component_scores"].values())
    assert score_rank_product(accesses, 1, None)["component_scores"] == dict(memory=None, underfill=None)


@pytest.mark.parametrize("architecture", ["sm_80", "sm_90a", "sm_100a", "sm_103", "sm_103a"])
def test_analysis_exports_portable_views_without_extra_models(monkeypatch, tmp_path, architecture):
    def forbidden(*args, **kwargs):
        pytest.fail("rank product must not add timing, bound classification, occupancy, or shared-memory analysis")

    for name in (
        "pipeline.analyze_pipeline",
        "occupancy.analyze_waves",
        "memory.resident_warps_estimate",
        "memory.analyze_compute_intensity",
        "shared_memory.analyze_shared_memory",
        "engine.predict_warp_specialization",
        "engine.select_specialization",
    ):
        monkeypatch.setattr(f"tilelang.tiletune.{name}", forbidden)
    kernel = gemm(stages=2)
    before = kernel.script()
    path = tmp_path / "rank-product.json"
    result = analyze_prim_func(
        kernel,
        dict(ranking_metric="rank_product", facts_path=str(path)),
        target=dict(kind="cuda", arch=architecture),
        device_limits=LIMITS,
    )
    facts = json.loads(path.read_text())
    expected = score_rank_product(facts["accesses"], facts["grid_blocks"], facts["sm_count"], facts["pipeline_depth"])
    assert kernel.script() == before
    assert facts["backend"] == "rank_product.v1"
    assert result["tile_cost"]["score"] is None
    assert result["tile_cost"]["component_scores"] == expected["component_scores"]
    assert result["tile_cost"]["score_scope"] == "candidate_pool"
    assert "bound" not in result["modules"]
    assert not result["modules"]["ranking"]["occupancy_gate_enabled"]
    assert result["modules"]["pipeline_overlap"]["precision"] == "disabled"
    assert result["modules"]["waves"]["precision"] == "disabled"


def test_unresolved_ledger_invalidates_both_components(monkeypatch):
    from tilelang.tiletune import memory

    analyze = memory.analyze_memory_accesses

    def unresolved(*args, **kwargs):
        result = analyze(*args, **kwargs)
        result["unknown"].append("opaque logical access")
        return result

    monkeypatch.setattr(memory, "analyze_memory_accesses", unresolved)
    result = analyze_prim_func(gemm(), dict(ranking_metric="rank_product"), target=dict(kind="cuda", arch="sm_103a"), device_limits=LIMITS)
    assert result["tile_cost"]["component_scores"] == dict(memory=None, underfill=None)
    assert result["tile_cost"]["precision"] == "unknown"
    assert rank_records([dict(index=0, **result)])[0]["tier"] == "unknown"


def test_runtime_freezes_pool_ranks_before_benchmarking(monkeypatch):
    records = records_for([(1, 1), (2, 6), (3, 3), (4, 4), (5, 2), (6, 5)])
    configs = [dict(identifier=index) for index in range(len(records))]
    session = TileTuneSession(TileTuneConfig(ranking_metric="rank_product", alpha=0.5, device_limits=LIMITS), configs)

    def analyze(program, *args, **kwargs):
        return dict(tile_cost=records[program]["tile_cost"], pressure=dict(decision=dict(keep=True)))

    monkeypatch.setattr("tilelang.tiletune.runtime.analyze_prim_func", analyze)
    selected = session.prepare_top_k([(index, config, {}) for index, config in enumerate(configs)], lambda identifier: identifier)
    assert selected == [0, 2, 4]
    assert set(session.prepared_programs) == set(selected)
    assert session.selection["requested_k"] == 3
    assert session.selection["strict_budget"]
    frozen = copy.deepcopy(session.ranking)
    session.benchmark_result(0, "ok", 100, None)
    session.benchmark_result(2, "ok", 1, None)
    report = session.finish()
    assert report["ranking"] == frozen
    assert report["selection"]["selected_indices"] == selected
    assert all(record["tile_cost"]["score"] is None for record in report["configs"])


def test_metric_is_opt_in_and_has_separate_cache_identity():
    assert TileTuneConfig().ranking_metric == "memory"
    keys = [TileTuneConfig(ranking_metric=metric).to_cache_key_dict() for metric in ("memory", "bound_aware", "rank_product")]
    assert len({json.dumps(key, sort_keys=True) for key in keys}) == 3
