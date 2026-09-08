"""A100 instruction identity, honest coverage, and portable experiment contracts."""

import json

import pytest

from tilelang.new_carver import CarverConfig, analyze_prim_func, load_device_profile, profile_device
from tilelang.new_carver import device_profile
from test_modules import PROFILE, attention
from test_pipeline import matrix_pipeline

AMPERE = {"kind": "cuda", "arch": "sm_80"}
# Explicit synthetic device limits: offline analysis must not query the local H200.
LIMITS = dict(
    sm_count=108,
    shared_memory_per_sm=167936,
    shared_memory_per_block=166912,
    registers_per_sm=65536,
    max_threads_per_sm=2048,
    max_threads_per_block=1024,
    max_blocks_per_sm=32,
    warp_size=32,
)


def ampere_profile():
    return {
        k: v
        for k, v in dict(
            PROFILE, profile_target="sm_80", gemm_signature=device_profile._signature("float16", "float32", "cuda.mma")
        ).items()
        if k != "wgmma_flops_per_cycle_per_warpgroup"
    }


@pytest.mark.parametrize("capability,arch", [((8, 0), "sm_80"), ((9, 0), "sm_90a")])
def test_current_target_respects_visible_device(monkeypatch, capability, arch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    assert device_profile.current_target()["arch"] == arch


def test_ampere_profile_cache_and_fp8_rejection(tmp_path, monkeypatch):
    identity = dict(profile_version=device_profile.PROFILE_VERSION, target_arch="sm_80")
    path = tmp_path / "a100.json"
    signature = device_profile._signature("float16", "float32", "cuda.mma")
    path.write_text(
        json.dumps(
            dict(
                identity=identity,
                common=dict(rates=PROFILE, clock_mhz=1400),
                gemm_models={json.dumps(signature, sort_keys=True): dict(rates={})},
            )
        )
    )
    monkeypatch.setattr(device_profile, "_identity", lambda: identity)

    def forbidden(*args):
        raise AssertionError("must not launch a profiling kernel")

    monkeypatch.setattr(device_profile, "_measure_common", forbidden)
    monkeypatch.setattr(device_profile, "_measure_gemm", forbidden)
    loaded = load_device_profile(path, input_dtype="float16")
    assert loaded["gemm_signature"]["instruction"] == "cuda.mma"
    assert profile_device(cache_path=path) == loaded
    with pytest.raises(ValueError, match="A100 has no FP8"):
        profile_device(cache_path=path, input_dtype="float8_e4m3fn")


@pytest.mark.parametrize("factory", [matrix_pipeline, attention])
@pytest.mark.parametrize("stages", [0, 2])
def test_ampere_analysis_keeps_unmodeled_software_pipeline(factory, stages):
    result = analyze_prim_func(
        factory(stages=stages),
        CarverConfig(enabled=True, mode="report_only", ranking_metric="pipeline_time", performance_model=ampere_profile()),
        target=AMPERE,
        device_limits=LIMITS,
    )
    assert result["pressure"]["warp_specialization"]["status"] == "not_applicable"
    pipeline = result["modules"]["pipeline_overlap"]
    assert all(p["compute_participants"]["instruction"] == "cuda.mma" for p in pipeline["phases"] if p["work"]["gemm_flops"])
    assert all(p["reduction"]["precision"] == "predicted" for p in pipeline["phases"] if p["reduction"])
    if stages:
        assert result["tile_cost"]["score"] is None
        assert "positive-stage pipeline scheduling policy is unresolved or unsupported" in pipeline["unknown"]
    else:
        assert result["tile_cost"]["score"] is not None
    assert result["pressure"]["decision"]["keep"]


def test_experiment_grids_and_unsupported_workloads():
    from benchmark.autotune.validate_new_carver_generalization import SPLITS, configs_for, unsupported_reason

    assert len(SPLITS["baseline"]) == 3
    assert len(SPLITS["all"]) == 17
    for w in SPLITS["all"]:
        assert len(configs_for(w)) == (288 if w["family"] == "gemm" else 128)
        assert bool(unsupported_reason(w, AMPERE)) == w["dtype"].startswith("float8")
    assert {w["causal"] for w in SPLITS["baseline"] if w["family"] == "attention"} == {True, False}


def test_experiment_freeze_includes_family_sources(tmp_path, monkeypatch):
    from benchmark.autotune import validate_new_carver_generalization as experiment
    from tilelang.cache.kernel_cache import KernelCache

    monkeypatch.chdir(experiment.REPO_ROOT)
    monkeypatch.setattr(experiment, "current_target", lambda: {"kind": "cuda", "arch": "sm_90a"})
    monkeypatch.setattr(experiment, "_identity", lambda: {})
    monkeypatch.setattr(experiment, "load_device_profile", lambda *args, **kwargs: {})
    monkeypatch.setattr("tilelang.new_carver.cost.query_device_limits", lambda target: LIMITS)
    monkeypatch.setattr(KernelCache, "_get_tilelang_lib_stamp", lambda: "test-native-build")
    profile = tmp_path / "profile.json"
    profile.write_text("{}")
    frozen = tmp_path / "frozen"
    experiment.freeze(frozen, profile, "baseline")
    report = experiment.verify(frozen)
    for relative in ("__init__.py", "families/__init__.py", "families/base.py", "families/gemm.py", "families/attention.py"):
        source = experiment.REPO_ROOT / "tilelang/new_carver" / relative
        key = f"tilelang/new_carver/{relative}"
        assert report["source_sha256"][key] == experiment.digest(source)
        assert (frozen / "sources" / key).read_bytes() == source.read_bytes()


def test_unknown_scores_do_not_claim_top_k_performance():
    from benchmark.autotune.new_carver_experiment_utils import evaluate_ranking
    from tilelang.new_carver import rank_records

    records = [dict(index=i, tile_cost=dict(score=None)) for i in range(2)]
    measured = [dict(index=i, config={}, benchmark_status="ok", latency_ms=i + 1) for i in range(2)]
    evaluation = evaluate_ranking(rank_records(records), records, measured)
    assert evaluation["winner"]["rank"] is None
    assert all(p["retained_percent"] is None for p in evaluation["prefix"].values())


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_ampere_probes_cross_compile(dtype):
    import tilelang
    from tilelang.new_carver.device_probes import tensor_core, tile_copy_roundtrip

    # Device compilation produces an sm_80 cubin without loading or executing it.
    from tvm.target import Target

    with Target(AMPERE):
        artifact = tilelang.lower(tensor_core(dtype, "float32", 3, 1, 128), target=AMPERE, enable_device_compile=True)
    assert "tl::mma_sync" in artifact.kernel_source
    assert "tl::wgmma" not in artifact.kernel_source
    with Target(AMPERE):
        artifact = tilelang.lower(tile_copy_roundtrip(3, 1, "sync"), target=AMPERE, enable_device_compile=True)
    assert "tl::tma_load" not in artifact.kernel_source


@pytest.mark.parametrize("mode,group_size", [("disabled", 1), ("report_only", 2), ("reject", 1)])
def test_gpu_portable_experiment_roundtrip(tmp_path, monkeypatch, mode, group_size):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from benchmark.autotune import validate_new_carver_generalization as experiment

    if torch.cuda.get_device_capability() not in ((8, 0), (9, 0)):
        pytest.skip("A100 or Hopper required")
    monkeypatch.chdir(experiment.REPO_ROOT)
    identity = device_profile._identity()
    signature = device_profile._signature("float16", "float32", device_profile._instruction(identity))
    path = tmp_path / "device.json"
    # Synthetic test rates exercise plumbing, never assert performance accuracy.
    path.write_text(
        json.dumps(
            dict(
                identity=identity,
                common=dict(rates=dict(PROFILE, dram_bytes_per_cycle=20), clock_mhz=1400),
                gemm_models={json.dumps(signature, sort_keys=True): dict(rates={})},
            )
        )
    )
    workloads = [
        dict(experiment.BASELINE[0], m=128, n=128, k=128),
        *[dict(w, batch=1, heads=1, query_length=256, kv_length=256, dim=64) for w in experiment.BASELINE[1:]],
    ]
    monkeypatch.setitem(experiment.SPLITS, "test", workloads)
    gemm = dict(block_M=64, block_N=64, block_K=32, thread_num=128, enable_rasteration=False)
    attn = dict(block_M=64, block_N=64, threads=128)
    monkeypatch.setattr(experiment, "configs_for", lambda w: [dict(gemm if w["family"] == "gemm" else attn, num_stages=s) for s in (0, 2)])
    root = tmp_path / "experiment"
    experiment.freeze(root, path, "test", mode=mode, group_size=group_size)
    for w in workloads:
        experiment.run(root, w)
        summary = json.loads((root / w["name"] / "summary.json").read_text())
        assert summary["error"] is None
        assert summary["record_errors"] == []
        assert len(summary["configs"]) == 2
        assert all(r["benchmark_status"] == "ok" for r in summary["configs"])
        assert bool(summary["carver"]) == (mode != "disabled")
        timing = (root / w["name"] / "timings.tsv").read_text()
        assert "duration_ms" in timing
        if mode != "disabled":
            assert "new_carver.analysis" in timing
    experiment.evaluate(root)
    assert len(json.loads((root / "evaluation.json").read_text())["results"]) == 3
    # A source edit must invalidate the frozen experiment.
    frozen = json.loads((root / "freeze.json").read_text())
    (root / "device.json").write_text("{}")
    assert frozen["source_sha256"]
    with pytest.raises(AssertionError):
        experiment.verify(root)
