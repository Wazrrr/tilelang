"""Shared analysis modules on GEMM and online-softmax attention IR."""

import pytest
from tilelang.new_carver import analyze_prim_func, CarverConfig
from tilelang.new_carver.pipeline import estimate_pipeline_cycles
from test_analysis import gemm
from test_cost import LIMITS


TARGET = {"kind": "cuda", "arch": "sm_90a"}
# Synthetic effective rates for testing the model, not measured H200 constants.
PROFILE = dict(
    global_bytes_per_cycle=64,
    shared_bytes_per_cycle=128,
    gemm_flops_per_cycle=2048,
    elementwise_ops_per_cycle=128,
    exp_ops_per_cycle=16,
    reduction_ops_per_cycle=64,
    reduction_local_sum_per_cycle=128,
    reduction_local_max_per_cycle=128,
    reduction_shuffle_sum_per_cycle=32,
    reduction_shuffle_max_per_cycle=32,
    copy_latency_cycles=400,
    barrier_cycles=16,
)
MODULES = {"register_pressure", "warp_specialization", "pipeline_overlap", "memory_traffic", "waves", "ranking"}


def attention(layout="bshd", causal=False, stages=2, sequence=128, block_n=64):
    if layout == "bshd":
        from examples.flash_attention.example_mha_fwd_bshd import flashattn

        shapes = dict(seq_len=sequence)
    else:
        from examples.flash_attention.example_mha_fwd_bhsd import flashattn

        shapes = dict(seq_q=sequence, seq_kv=sequence)
    return flashattn.jit_impl.get_tir(
        batch=1, heads=2, dim=64, is_causal=causal, block_M=64, block_N=block_n, num_stages=stages, threads=128, **shapes
    )


@pytest.mark.parametrize("family", ["gemm", "attention"])
def test_same_modules_and_unchanged_ir(family):
    func = gemm(stages=2) if family == "gemm" else attention()
    before = func.script()
    result = analyze_prim_func(func, target=TARGET, device_limits=LIMITS)
    assert func.script() == before
    assert result["specialization"]["name"] == family
    assert set(result["modules"]) == MODULES
    assert all("implementation" in module for module in result["modules"].values())
    assert result["modules"]["register_pressure"] == result["pressure"]
    assert result["modules"]["warp_specialization"]["status"] == "predicted"
    assert result["modules"]["ranking"]["score"] == result["tile_cost"]["score"]
    assert result["modules"]["pipeline_overlap"]["timing_status"] == "unknown"  # no invented timing profile


@pytest.mark.parametrize("layout", ["bshd", "bhsd"])
@pytest.mark.parametrize("causal", [False, True])
def test_attention_traffic_liveness_and_causal_bounds(layout, causal):
    result = analyze_prim_func(attention(layout, causal), target=TARGET, device_limits=LIMITS)
    assert result["specialization"]["name"] == "attention"
    assert not result["tile_propagation"]["unknown"]
    memory = result["modules"]["memory_traffic"]
    assert len(memory["input_tiles"]) == 3  # Q once, K and V once each per iteration
    assert memory["one_time_input_bytes"] == 64 * 64 * 2
    assert memory["per_iteration_input_bytes"] == 2 * 64 * 64 * 2
    assert memory["input_bytes_per_block"] == (1 + 2 * 2) * 64 * 64 * 2
    assert memory["output_bytes_per_block"] == 64 * 64 * 2
    assert memory["cta_work_uniform"] == (not causal)
    allocations = {a["buffer"]: a for a in memory["shared_allocations"]}
    assert allocations["Q_shared"]["pipeline_copies_estimate"] == allocations["O_shared"]["pipeline_copies_estimate"] == 1
    assert allocations["K_shared"]["pipeline_copies_estimate"] == allocations["V_shared"]["pipeline_copies_estimate"] == 2
    state = result["pressure"]["tile_liveness"]
    assert {"acc_o", "logsum", "scores_max"} <= set(state["loop_carried_buffers"])
    score_phase = next(p for p in state["phases"] if p["phase"] == "qk_gemm")
    assert {"acc_s", "acc_o"} <= {b["buffer"] for b in score_phase["buffers"]}
    assert state["peak_registers_per_block_estimate"] > result["pressure"]["modeled_accumulator_registers_per_block"]
    pipe = result["modules"]["pipeline_overlap"]
    assert pipe["iterations"]["max"] == 2
    assert pipe["iterations"]["min"] == (1 if causal else 2)
    assert pipe["overlap_eligible"]
    assert sum(p["work"]["exp_ops"] for p in pipe["phases"]) > 0
    assert sum(p["work"]["reduction_ops"] for p in pipe["phases"]) > 0
    assert result["tile_cost"]["score"] is not None


def test_pipeline_stages_help_only_through_modeled_overlap():
    results = [
        analyze_prim_func(
            gemm(stages=s, extent=16),
            {"ranking_metric": "pipeline_time", "performance_model": PROFILE},
            target=TARGET,
            device_limits=LIMITS,
        )
        for s in (1, 2, 3)
    ]
    times = [r["modules"]["pipeline_overlap"]["timing"]["cycles"] for r in results]
    assert times[0] > times[1] > times[2]
    assert len({r["pressure"]["modeled_lower_bound"] for r in results}) == 1
    assert len({r["tile_cost"]["traffic_bytes_per_block"] for r in results}) == 1
    assert all(r["modules"]["ranking"]["units"] == "cycles" for r in results)
    pipe = results[1]["modules"]["pipeline_overlap"]
    one, two = [estimate_pipeline_cycles(pipe, concurrent_ctas=n) for n in (1, 2)]
    assert two["copy_service_cycles_per_iteration"] == 2 * one["copy_service_cycles_per_iteration"]
    assert two["consumer_cycles_per_iteration"] > one["consumer_cycles_per_iteration"]
    assert two["cycles"] >= one["cycles"]


def test_attention_timing_contains_both_gemms_softmax_and_once_only_query():
    result = analyze_prim_func(
        attention(stages=3), {"ranking_metric": "pipeline_time", "performance_model": PROFILE}, target=TARGET, device_limits=LIMITS
    )
    pipe = result["modules"]["pipeline_overlap"]
    gemms = [p for p in pipe["phases"] if p["work"]["gemm_flops"]]
    assert [p["phase"] for p in gemms] == ["qk_gemm", "pv_gemm"]
    assert all(p["work"]["gemm_flops"] == 2 * 64**3 for p in gemms)
    assert pipe["producer_copies_per_iteration"] == 2
    assert pipe["outside_loop_input_copies"] == 1
    assert pipe["timing"] is not None
    assert result["tile_cost"]["score"] > 0


def test_profiles_and_specializations_are_in_cache_identity():
    default = CarverConfig().to_cache_key_dict()
    assert default != CarverConfig(specialization="attention").to_cache_key_dict()
    assert default != CarverConfig(ranking_metric="pipeline_time", performance_model=PROFILE).to_cache_key_dict()
    assert CarverConfig(report_path="one").to_cache_key_dict() == CarverConfig(report_path="two").to_cache_key_dict()
    for profile in ({"unknown": 1}, {"global_bytes_per_cycle": 0}, {"copy_latency_cycles": float("nan")}):
        with pytest.raises(ValueError):
            CarverConfig(performance_model=profile)


def test_unmatched_specialization_and_missing_profile_remain_eligible():
    for config in (
        {"specialization": "attention"},
        {"ranking_metric": "pipeline_time"},
        {"ranking_metric": "pipeline_time", "performance_model": {"global_bytes_per_cycle": 64}},
    ):
        result = analyze_prim_func(gemm(stages=2), config, target=TARGET, device_limits=LIMITS)
        assert result["pressure"]["decision"]["keep"]
        assert result["tile_cost"]["score"] is None


def test_pipeline_errors_reach_the_caller(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("unknown schedule")

    monkeypatch.setattr("tilelang.new_carver.pipeline.analyze_pipeline", fail)
    with pytest.raises(RuntimeError, match="unknown schedule"):
        analyze_prim_func(attention(), target=TARGET, device_limits=LIMITS)


def test_different_score_units_cannot_be_mixed():
    from tilelang.new_carver import rank_records

    with pytest.raises(ValueError, match="different ranking metrics"):
        rank_records(
            [
                {"index": 0, "tile_cost": {"score": 1, "ranking_metric": "traffic_waves"}},
                {"index": 1, "tile_cost": {"score": 1, "ranking_metric": "pipeline_time"}},
            ]
        )


def test_unrelated_gemms_do_not_select_attention():
    import tilelang.language as T

    @T.prim_func
    def main(X: T.Tensor((64, 64), "float16"), Y: T.Tensor((64, 64), "float16"), Out: T.Tensor((64, 64), "float32")):
        with T.Kernel(1, threads=128):
            x = T.alloc_shared((64, 64), "float16")
            y = T.alloc_shared((64, 64), "float16")
            first = T.alloc_fragment((64, 64), "float32")
            second = T.alloc_fragment((64, 64), "float32")
            for _k in T.Pipelined(2, num_stages=2):
                T.copy(X, x)
                T.copy(Y, y)
                T.gemm(x, y, first, clear_accum=True)
                T.gemm(x, y, second, clear_accum=True)
            T.copy(second, Out)

    assert analyze_prim_func(main, target=TARGET)["specialization"]["name"] == "generic"


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
def test_gpu_attention_modules_and_correctness(tmp_path, causal, grouped):
    import json
    import re
    import torch
    from tvm.target import Target
    from tilelang.autotuner.grouped_compile import compile_grouped_unit_tvm_ffi
    from tilelang.autotuner.param import CompileArgs
    from tilelang.new_carver.runtime import CarverSession

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Hopper required")
    configs = [{"block_n": 64, "stages": 1}, {"block_n": 128, "stages": 2}]
    session = CarverSession(
        CarverConfig(
            enabled=True,
            mode="report_only",
            report_path=str(tmp_path / "attention.json"),
            ranking_metric="pipeline_time",
            performance_model=PROFILE,
            device_limits=LIMITS,
        ),
        configs,
    )
    calls = []

    def elaborate(block_n, stages):
        calls.append((block_n, stages))
        return attention(causal=causal, stages=stages, block_n=block_n)

    items = list(enumerate(configs))
    results = []
    for unit in [items] if grouped else [[item] for item in items]:
        results.extend(
            compile_grouped_unit_tvm_ffi(
                unit,
                CompileArgs(
                    target=Target(TARGET),
                    out_idx=[3],
                    execution_backend="tvm_ffi",
                    pass_configs={"tl.enable_cuda_resource_capture": True},
                ),
                elaborate,
                carver_session=session,
            )
        )
    assert calls == [(64, 1), (128, 2)]
    torch.manual_seed(11)
    q, k, v = [torch.randn((1, 128, 2, 64), dtype=torch.float16, device="cuda") for _ in range(3)]
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / 8
    if causal:
        scores.masked_fill_(torch.ones((128, 128), device="cuda", dtype=torch.bool).triu(1), -float("inf"))
    reference = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v.float())
    for index, _, jit, error in results:
        assert error is None, str(error)
        record = session.records[index]
        assert record["specialization"]["name"] == "attention"
        assert set(record["modules"]) == MODULES
        policy = record["modules"]["warp_specialization"]
        assert policy["status"] == "predicted", policy
        name = next(iter(record["compiler_resources"]))
        entry = re.search(r"__launch_bounds__\((\d+),[^)]*\)\s+" + re.escape(name) + r"\([^\n]*\{[\s\S]*?\n\}", jit.get_kernel_source())
        assert entry is not None and int(entry[1]) == policy["launch_threads"]
        assert f"warpgroup_reg_alloc<{policy['consumer_register_request']}>" in entry[0]
        torch.testing.assert_close(jit(q, k, v).float(), reference, rtol=0.02, atol=0.02)
        assert record["tile_cost"]["score"] is not None
        assert record["post_compile"]["status"] == "pass"
    report = session.finish()
    assert len(report["ranking"]) == 2
    assert len(json.loads((tmp_path / "attention.json").read_text())["configs"]) == 2


def test_gpu_attention_autotuner_reports_every_module(tmp_path):
    import json
    import torch
    from tvm.target import Target
    from tilelang.autotuner import AutoTuner

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Hopper required")
    calls = []

    def kernel_factory(stages):
        return attention(stages=stages)

    def elaborate(stages):
        calls.append(stages)
        return kernel_factory(stages)

    def reference(q, k, v):
        scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / 8
        return torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v.float()).to(q.dtype)

    path = tmp_path / "tuning.json"
    tuner = (
        AutoTuner(kernel_factory, [{"stages": 1}, {"stages": 2}])
        .set_compile_args(target=Target(TARGET), out_idx=[3], execution_backend="tvm_ffi")
        .set_profile_args(ref_prog=reference, rtol=0.02, atol=0.02)
        .set_carver_args(True, mode="report_only", report_path=str(path), ranking_metric="pipeline_time", performance_model=PROFILE)
    )
    tuner.jit_elaborate = elaborate
    result = tuner.run(warmup=1, rep=2, enable_grouped_compile=True, group_compile_size=2)
    assert result.kernel is not None
    assert sorted(calls) == [1, 2]
    report = json.loads(path.read_text())
    assert len(report["configs"]) == len(report["ranking"]) == 2
    assert all(r["status"] == "benchmarked" and set(r["modules"]) == MODULES for r in report["configs"])
    assert all(r["modules"]["ranking"]["metric"] == "pipeline_time" for r in report["configs"])
