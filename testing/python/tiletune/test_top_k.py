"""Top-k must be frozen before compilation and retain the full candidate record."""

import json

import pytest
import torch

from tilelang.autotuner import AutoTuner
from tilelang.tiletune import TileTuneConfig
from tilelang.tiletune.ranking import rank_records, select_top_k
from tilelang.tiletune.runtime import TileTuneSession
from test_integration import kernel


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_invalid_top_k(value):
    with pytest.raises(ValueError, match="top_k"):
        TileTuneConfig(top_k=value)


def test_top_k_requires_ranking_and_changes_cache_identity():
    with pytest.raises(ValueError, match="ranking=True"):
        TileTuneConfig(top_k=2, ranking=False)
    keys = [TileTuneConfig(top_k=k).to_cache_key_dict() for k in (None, 1, 2)]
    assert keys[0] != keys[1] != keys[2]
    with pytest.raises(ValueError, match="requires top_k"):
        TileTuneConfig(strict_top_k=True)
    with pytest.raises(ValueError, match="does not support exploration"):
        TileTuneConfig(top_k=2, strict_top_k=True, exploration_fraction=0.5)


def test_top_k_ties_unknowns_and_shortfall():
    records = [dict(index=i, tile_cost={"score": score}) for i, score in enumerate([None, 20, 10, 10, float("inf")])]
    records.append(dict(index=5, tile_cost={"score": 1}, pre_lowering={"would_reject": True}))
    ranking = rank_records(records)
    assert select_top_k(ranking, 1) == [2, 3]
    assert select_top_k(ranking, 1, include_ties=False) == [2]
    assert [row["rank"] for row in ranking[:3]] == [2, 2, 3]
    assert select_top_k(ranking, 8) == [2, 3, 1, 0, 4]
    for record in records:
        record["latency_ms"] = 1 / (record["index"] + 1)
    assert select_top_k(rank_records(records), 2) == [2, 3]


def test_strict_top_k_drops_the_complete_boundary_group():
    records = [
        dict(index=i, status="analyzed", tile_cost={"score": score})
        for i, score in enumerate([10, 10, 20, None, None])
    ]
    ranking = rank_records(records)
    assert select_top_k(ranking, 1, strict_budget=True) == []
    assert select_top_k(ranking, 2, strict_budget=True) == [0, 1]
    assert select_top_k(ranking, 4, strict_budget=True) == [0, 1, 2]
    assert select_top_k(ranking, 5, strict_budget=True) == [0, 1, 2, 3, 4]


def test_preparation_keeps_the_whole_boundary_tie(monkeypatch):
    session = TileTuneSession(TileTuneConfig(top_k=1), [{"id": i} for i in range(3)])

    def analyze(program, *args, **kwargs):
        return dict(tile_cost={"score": 10}, pressure={"decision": {"keep": True}})

    monkeypatch.setattr("tilelang.tiletune.runtime.analyze_prim_func", analyze)
    selected = session.prepare_top_k([(i, {"id": i}, {}) for i in range(3)], lambda id: id)
    assert selected == [0, 1, 2]
    assert set(session.prepared_programs) == {0, 1, 2}
    assert session.selection["requested_k"] == 1
    assert session.selection["selected_count"] == 3
    assert session.selection["budget_excess"] == 2
    assert session.selection["tie_policy"] == "include_boundary_score_group"


def test_strict_preparation_drops_a_boundary_tie(monkeypatch):
    session = TileTuneSession(TileTuneConfig(top_k=2, strict_top_k=True), [{"id": i} for i in range(4)])

    def analyze(program, *args, **kwargs):
        return dict(tile_cost={"score": 10 if program < 3 else 20}, pressure={"decision": {"keep": True}})

    monkeypatch.setattr("tilelang.tiletune.runtime.analyze_prim_func", analyze)
    selected = session.prepare_top_k([(i, {"id": i}, {}) for i in range(4)], lambda id: id)
    assert selected == []
    assert session.selection["selected_count"] == 0
    assert session.selection["shortfall"] == 2
    assert session.selection["budget_excess"] == 0
    assert session.selection["strict_budget"]
    assert session.selection["tie_policy"] == "exclude_boundary_score_group"


@pytest.mark.parametrize("alpha", [0, -1, 1.1, True, float("inf"), float("nan")])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        TileTuneConfig(alpha=alpha)


def test_alpha_uses_original_pool_and_excludes_unknowns_and_crossing_ties(monkeypatch):
    session = TileTuneSession(TileTuneConfig(alpha=0.5), [{"id": i} for i in range(5)])

    def analyze(program, *args, **kwargs):
        if program == 4:
            raise ValueError("opaque candidate")
        return dict(tile_cost={"score": 1 if program == 0 else 2}, pressure={"decision": {"keep": True}})

    monkeypatch.setattr("tilelang.tiletune.runtime.analyze_prim_func", analyze)
    assert session.prepare_top_k([(i, {"id": i}, {}) for i in range(5)], lambda id: id) == [0]
    assert session.selection["requested_k"] == 2
    assert session.selection["pool_size"] == 5
    assert session.selection["alpha"] == 0.5
    assert session.selection["shortfall"] == 1
    assert session.selection["budget_excess"] == 0
    assert session.selection["strict_budget"]
    assert session.records[-1]["status"] == "analysis_failed"


def test_alpha_configuration_contract_and_cache_identity():
    for settings in (
        dict(alpha=0.5, top_k=2),
        dict(alpha=0.5, ranking=False),
        dict(alpha=0.5, exploration_fraction=0.5),
        dict(strict_top_k=True),
        dict(top_k=2, strict_top_k=1),
    ):
        with pytest.raises(ValueError):
            TileTuneConfig(**settings)
    assert TileTuneConfig(alpha=0.5).to_cache_key_dict() != TileTuneConfig(top_k=2).to_cache_key_dict()
    with pytest.raises(ValueError, match="selects no candidates"):
        TileTuneSession(TileTuneConfig(alpha=0.1), [{"id": 0}])


def test_preparation_retains_failures_and_never_refills(monkeypatch):
    session = TileTuneSession(TileTuneConfig(top_k=2), [{"id": i} for i in range(4)])
    calls = []

    def elaborate(id):
        calls.append(id)
        if id == 0:
            raise ValueError("invalid configuration")
        return id

    def analyze(program, *args, **kwargs):
        return dict(tile_cost={"score": 10 if program == 3 else None}, pressure={"decision": {"keep": True}})

    monkeypatch.setattr("tilelang.tiletune.runtime.analyze_prim_func", analyze)
    chosen = session.prepare_top_k([(i, cfg, {}) for i, cfg in enumerate([{"id": i} for i in range(4)])], elaborate)
    assert chosen == [3, 1, 2]
    assert calls == [0, 1, 2, 3]
    assert session.elaborate(3, {"id": 3}, elaborate) == 3
    assert calls == [0, 1, 2, 3]
    session.compilation_result(3, RuntimeError("compiler failed"))
    result = session.finish()
    assert result["selection"]["shortfall"] == 0
    assert result["selection"]["budget_excess"] == 1
    assert result["selection"]["selected_indices"] == [3, 1, 2]
    assert [r["status"] for r in result["configs"]] == ["elaboration_failed", "analyzed", "analyzed", "compilation_failed"]


def test_permitted_unknown_boundary_is_retained_but_unavailable_is_not():
    records = [
        dict(index=0, status="analyzed", tile_cost={"score": 1}),
        dict(index=1, status="analyzed", tile_cost={"score": None}),
        dict(index=2, status="analyzed", tile_cost={"score": None}),
        dict(index=3, status="analysis_failed", tile_cost={"score": None}),
    ]
    ranking = rank_records(records)
    assert [row["tier"] for row in ranking] == ["eligible", "unknown", "unknown", "unavailable"]
    assert [row["rank"] for row in ranking] == [1, 3, 3, 4]
    assert select_top_k(ranking, 1) == [0]
    assert select_top_k(ranking, 2) == [0, 1, 2]


def scored_kernel(block=32, score=10):
    return kernel(block).with_attr("test_score", score)


@pytest.mark.parametrize("grouped,override", [(False, True), (True, True), (True, False)])
@pytest.mark.parametrize("fail_first", [False, True])
def test_gpu_top_k_reuses_ir_and_compiles_only_selected_indices(monkeypatch, tmp_path, grouped, override, fail_first):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from tilelang.tiletune import runtime
    from tilelang.autotuner import grouped_compile

    original_analysis = runtime.analyze_prim_func
    original_lower = grouped_compile.lower_to_host_device_ir
    analyzed, lowered, elaborated = [], [], []
    configs = [
        {"block": 32, "score": 30},
        {"block": 64, "score": 10},
        {"block": 32, "score": 20, "pass_configs": {"tl.disable_wgmma": True}},
    ]
    if not override:
        configs[2].pop("pass_configs")

    def analyze(program, *args, **kwargs):
        analyzed.append(int(program.attrs["test_score"]))
        result = original_analysis(program, *args, **kwargs)
        result["tile_cost"]["score"] = int(program.attrs["test_score"])
        return result

    def lower(program, **kwargs):
        assert analyzed == [30, 10, 20]
        saved = json.loads((tmp_path / "report.json").read_text())
        assert saved["selection"]["selected_indices"] == [1, 2]
        score = int(program.attrs["test_score"])
        lowered.append(score)
        if fail_first and score == 10:
            raise RuntimeError("deliberate selected compilation failure")
        return original_lower(program, **kwargs)

    def elaborate(**kwargs):
        kwargs.pop("__pass_configs__", None)
        elaborated.append(kwargs["score"])
        return scored_kernel(**kwargs)

    monkeypatch.setattr(runtime, "analyze_prim_func", analyze)
    monkeypatch.setattr(grouped_compile, "lower_to_host_device_ir", lower)
    tuner = (
        AutoTuner(scored_kernel, configs)
        .set_compile_args(target="cuda", execution_backend="tvm_ffi", out_idx=[2])
        .set_profile_args(ref_prog=lambda a, b: a.float() @ b.float(), rtol=0.01, atol=0.01)
        .set_tiletune_args(True, top_k=2, mode="report_only", report_path=str(tmp_path / "report.json"))
    )
    tuner.jit_elaborate = elaborate
    result = tuner.run(warmup=1, rep=2, use_pipeline=True, enable_grouped_compile=grouped, group_compile_size=2)
    assert result.config in configs[1:]
    assert elaborated == analyzed == [30, 10, 20]
    assert sorted(lowered) == [10, 20]
    report = tuner.tiletune_report
    assert report["selection"]["selected_indices"] == [1, 2]
    assert report["configs"][0]["status"] == "not_selected"
    assert report["configs"][1]["status"] == ("compilation_failed" if fail_first else "benchmarked")
    assert report["configs"][2]["status"] == "benchmarked"
    if override:
        assert report["configs"][2]["effective_pass_configs"]["tl.disable_wgmma"]
