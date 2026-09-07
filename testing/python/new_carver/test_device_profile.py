"""Device profiles transfer by instruction/dtype; one sample cannot fit a ranking."""

import json

import pytest

from tilelang.new_carver import analyze_prim_func, anchor_latency, load_device_profile, profile_device, rank_records, CarverConfig
from tilelang.new_carver import device_profile
from test_cost import LIMITS
from test_modules import PROFILE, TARGET, attention
from test_pipeline import matrix_pipeline


def profile(dtype="float16"):
    return dict(
        PROFILE,
        reference_clock_mhz=1800,
        profile_target="sm_90a",
        profile_id="synthetic-test-profile",
        gemm_signature=dict(instruction="cuda.wgmma", a_dtype=dtype, b_dtype=dtype, accum_dtype="float32"),
    )


def analyze(func, rates):
    return analyze_prim_func(func, dict(ranking_metric="pipeline_time", performance_model=rates), target=TARGET, device_limits=LIMITS)


def test_one_latency_anchor_preserves_order_and_matches_reference():
    rates = profile()
    funcs = [matrix_pipeline(stages=s) for s in (0, 1, 2, 3)]
    before = [dict(index=i, **analyze(f, rates)) for i, f in enumerate(funcs)]
    anchored = anchor_latency(rates, before[2], 0.2)
    after = [dict(index=i, **analyze(f, anchored)) for i, f in enumerate(funcs)]
    assert rates == profile()
    assert after[2]["modules"]["ranking"]["estimated_latency_ms"] == pytest.approx(0.2)
    assert [r["index"] for r in rank_records(before)] == [r["index"] for r in rank_records(after)]
    assert all(b["pressure"] == a["pressure"] for b, a in zip(before, after))
    reanchored = anchor_latency(anchored, after[2], 0.3)
    assert analyze(funcs[2], reanchored)["modules"]["ranking"]["estimated_latency_ms"] == pytest.approx(0.3)


@pytest.mark.parametrize("latency", [0, -1, float("nan"), float("inf")])
def test_invalid_anchor(latency):
    with pytest.raises(ValueError, match="positive"):
        anchor_latency(profile(), {}, latency)


def test_anchor_requires_exact_profile_and_score():
    rates = profile()
    result = analyze(matrix_pipeline(), rates)
    with pytest.raises(ValueError, match="exact profile"):
        anchor_latency(dict(rates, barrier_cycles=100), result, 0.1)
    with pytest.raises(ValueError, match="exact profile"):
        anchor_latency(rates, analyze(matrix_pipeline(), profile("float8_e4m3fn")), 0.1)


def test_device_profile_dtype_and_instruction_guards_preserve_pressure():
    func = matrix_pipeline()
    for rates in [profile("float8_e4m3fn"), dict(profile(), profile_target="sm_80")]:
        result = analyze(func, rates)
        assert result["tile_cost"]["score"] is None
        assert not result["pressure"]["decision"]["would_reject"]
        assert result["modules"]["pipeline_overlap"]["unknown"]
    result = analyze_prim_func(
        func,
        dict(ranking_metric="pipeline_time", performance_model=profile()),
        target=TARGET,
        device_limits=LIMITS,
        pass_configs={"tl.disable_wgmma": True},
    )
    assert result["tile_cost"]["score"] is None


def test_same_profile_handles_attention_work():
    result = analyze(attention(stages=3), profile())
    assert result["tile_cost"]["score"] is not None
    assert any(p["work"]["exp_ops"] for p in result["modules"]["pipeline_overlap"]["phases"])


@pytest.mark.parametrize("change", ["legacy", "dtype"])
def test_reduction_profile_mismatch_retains_candidate(change):
    rates = profile()
    if change == "legacy":
        rates = {k: v for k, v in rates.items() if not k.startswith(("reduction_local_", "reduction_shuffle_"))}
    else:
        rates["reduction_dtype"] = "float16"
    result = analyze(attention(), rates)
    assert result["tile_cost"]["score"] is None
    assert result["pressure"]["decision"]["keep"]
    assert any("reduction" in reason for reason in result["modules"]["pipeline_overlap"]["unknown"])


def test_old_device_profile_requires_new_primitive_measurements(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps(dict(identity=dict(profile_version=1))))
    with pytest.raises(ValueError, match="regenerate"):
        load_device_profile(path, input_dtype="float16")


def test_cached_profile_requires_no_gpu_or_benchmark(tmp_path, monkeypatch):
    path = tmp_path / "device.json"
    identity = dict(profile_version=device_profile.PROFILE_VERSION, target_arch="sm_90a", device_name="test")
    signature = device_profile._signature("float8_e4m3fn", "float32")
    data = dict(
        identity=identity, common=dict(rates=PROFILE, clock_mhz=1800), gemm_models={json.dumps(signature, sort_keys=True): dict(rates={})}
    )
    path.write_text(json.dumps(data))
    monkeypatch.setattr(device_profile, "_identity", lambda: identity)

    def forbidden(*args, **kwargs):
        raise AssertionError("cached profile must not measure")

    monkeypatch.setattr(device_profile, "_measure_common", forbidden)
    monkeypatch.setattr(device_profile, "_measure_gemm", forbidden)
    loaded = load_device_profile(path, input_dtype="float8_e4m3fn")
    assert profile_device(input_dtype="float8_e4m3fn", cache_path=path) == loaded
    monkeypatch.setattr(device_profile, "_identity", forbidden)
    assert load_device_profile(path, input_dtype="float8_e4m3fn") == loaded
    with pytest.raises(ValueError, match="no measurements"):
        load_device_profile(path, input_dtype="float16")
    with pytest.raises(ValueError, match="fingerprint"):
        load_device_profile(path, input_dtype="float8_e4m3fn", expected_identity={})
    CarverConfig(performance_model=loaded)


def test_missing_dtype_only_measures_dtype_primitives(tmp_path, monkeypatch):
    path = tmp_path / "device.json"
    identity = dict(profile_version=device_profile.PROFILE_VERSION, target_arch="sm_90a")
    monkeypatch.setattr(device_profile, "_identity", lambda: identity)
    common_calls, gemm_calls = [], []

    def common(_identity):
        common_calls.append(True)
        return dict(rates=dict(PROFILE, dram_bytes_per_cycle=12), clock_mhz=1800)

    def gemm(_identity, clock, dtype, accum_dtype):
        gemm_calls.append(dtype)
        return dict(rates=dict(gemm_flops_per_cycle=4096 if dtype == "float8_e4m3fn" else 2048))

    monkeypatch.setattr(device_profile, "_measure_common", common)
    monkeypatch.setattr(device_profile, "_measure_gemm", gemm)
    for dtype in ("float8_e4m3fn", "float16", "float8_e4m3fn"):
        profile_device(input_dtype=dtype, cache_path=path)
    assert len(common_calls) == 1
    assert gemm_calls == ["float8_e4m3fn", "float16"]
    streaming = load_device_profile(path, input_dtype="float16", memory_regime="streaming")
    assert streaming["global_bytes_per_cycle"] == 12
    assert streaming["memory_regime"] == "streaming"
    CarverConfig(performance_model=streaming)
    monkeypatch.setattr(device_profile, "_identity", lambda: dict(identity, device_name="different"))
    with pytest.raises(ValueError, match="fingerprint"):
        profile_device(cache_path=path)
    assert len(common_calls) == 1
    with pytest.raises(ValueError, match="memory_regime"):
        profile_device(cache_path=path, memory_regime="unspecified")


def test_fp8_example_uses_actual_ir_and_fp32_accumulation():
    from examples.gemm_fp8.example_gemm_fp8_new_carver import get_configs, make_gemm

    configs = get_configs()
    assert len(configs) == 288
    factory = make_gemm(4096, 4096, 4096, "float8_e4m3fn")
    result = analyze(factory(**configs[157]), profile("float8_e4m3fn"))
    assert result["tile_cost"]["score"] is not None
    memory = result["modules"]["memory_traffic"]
    assert memory["per_iteration_input_bytes"] == (128 * 64 + 128 * 64)
    assert memory["output_bytes_per_block"] == 128 * 128
    assert result["pressure"]["modeled_lower_bound"] == 128


@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("grouped", [False, True])
def test_gpu_fp8_example_elaborates_once_with_inferred_outputs(dtype, grouped):
    import torch
    import tilelang.language as T
    from tvm.target import Target
    from tilelang.autotuner.grouped_compile import compile_grouped_unit_tvm_ffi
    from tilelang.autotuner.param import CompileArgs
    from tilelang.new_carver.runtime import CarverSession
    from examples.gemm_fp8.example_gemm_fp8_new_carver import get_configs, make_gemm
    from examples.gemm_fp8.example_tilelang_gemm_fp8 import calc_diff

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Hopper required")
    configs = [get_configs()[29], get_configs()[157]]
    session = CarverSession(
        CarverConfig(
            enabled=True, mode="report_only", device_limits=LIMITS, ranking_metric="pipeline_time", performance_model=profile(dtype)
        ),
        configs,
    )
    calls = []
    factory = make_gemm(256, 256, 256, dtype)

    def elaborate(**config):
        calls.append(config)
        return factory(**config)

    results = []
    items = list(enumerate(configs))
    for unit in [items] if grouped else [[item] for item in items]:
        results.extend(
            compile_grouped_unit_tvm_ffi(
                unit, CompileArgs(target=Target(TARGET), execution_backend="tvm_ffi"), elaborate, carver_session=session
            )
        )
    assert calls == configs
    torch.manual_seed(17)
    inputs = [torch.randn((256, 256), device="cuda", dtype=torch.float16).to(T.dtype(dtype).as_torch()) for _ in range(2)]
    reference = (inputs[0].float() @ inputs[1].float().T).to(inputs[0].dtype)
    for index, _config, kernel, error in results:
        assert error is None, str(error)
        assert calc_diff(kernel(*inputs), reference).item() < 1e-3
        assert session.records[index]["tile_cost"]["score"] is not None
        assert session.records[index]["post_compile"]["status"] == "pass"
        assert session.records[index]["compiler_resources"]


def test_gpu_wgmma_probe_has_nonzero_input_dependent_odd_iteration_result():
    import torch
    from tilelang.new_carver.device_profile import _benchmark
    from tilelang.new_carver.device_probes import tensor_core

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Hopper required")
    a = torch.full((64, 128), 0.125, device="cuda").to(torch.float8_e4m3fn)
    b = torch.full((128, 128), 0.25, device="cuda").to(torch.float8_e4m3fn)
    # Throughput probes use paired even counts. An odd count must retain the
    # input-dependent product, ruling out a silently zeroed/no-op probe.
    for iterations in (127, 255):
        _, evidence, output = _benchmark(
            tensor_core("float8_e4m3fn", "float32", iterations, 1, 128), [a, b], [2], required_source="tl::wgmma_ss"
        )
        torch.testing.assert_close(output, torch.full_like(output, -4.0))
        assert evidence["required_instruction_verified"]


@pytest.mark.parametrize("kind", [0, 1, 2, 3])
def test_gpu_reduction_primitives_execute_local_or_lane_work(kind):
    import torch
    import tilelang
    from tilelang.new_carver.device_probes import reduction_primitive

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    kernel = tilelang.compile(reduction_primitive(kind, 3, 1), out_idx=[0], execution_backend="tvm_ffi")
    initial = 0.01 * (torch.arange(128, device="cuda")[:, None] + torch.arange(8, device="cuda")[None, :] + 1)
    partner = initial[torch.arange(128, device="cuda") ^ 1]
    expected = [initial + 0.0003, initial.clamp_min(0.5), (initial + partner) / 2, torch.maximum(initial, partner)][kind]
    torch.testing.assert_close(kernel()[0], expected.sum(1), rtol=1e-5, atol=1e-6)
