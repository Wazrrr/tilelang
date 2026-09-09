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


def test_top_k_ties_unknowns_and_shortfall():
    records = [dict(index=i, tile_cost={"score": score}) for i, score in enumerate([None, 20, 10, 10, float("inf")])]
    records.append(dict(index=5, tile_cost={"score": 1}, pre_lowering={"would_reject": True}))
    ranking = rank_records(records)
    assert select_top_k(ranking, 1) == [2]
    assert select_top_k(ranking, 8) == [2, 3, 1]
    for record in records:
        record["latency_ms"] = 1 / (record["index"] + 1)
    assert select_top_k(rank_records(records), 2) == [2, 3]


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
    assert chosen == [3]
    assert calls == [0, 1, 2, 3]
    assert session.elaborate(3, {"id": 3}, elaborate) == 3
    assert calls == [0, 1, 2, 3]
    session.compilation_result(3, RuntimeError("compiler failed"))
    result = session.finish()
    assert result["selection"]["shortfall"] == 1
    assert result["selection"]["selected_indices"] == [3]
    assert [r["status"] for r in result["configs"]] == ["elaboration_failed", "not_selected", "not_selected", "compilation_failed"]


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
