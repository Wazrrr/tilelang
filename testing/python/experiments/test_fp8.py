"""Native FP8 experiment contracts, quantization checks and offline lowering."""

from dataclasses import replace

import pytest

from experiments.common.spec import Device, TARGETS, support_reason
from experiments.gemm_fp8.cases import cases


def test_fp8_example_output_attributes_reach_the_compile_adapter(monkeypatch):
    import importlib
    import tilelang
    from experiments.gemm_fp8.kernel import make_case

    case = make_case(cases()[0])
    func = case.build(block_M=128, block_N=128, block_K=64, num_stages=2, threads=128, enable_rasteration=False)
    inferred = list(func.attrs["tilelang_out_idx"])
    assert [int(i) % len(func.params) for i in inferred] == [2]
    assert case.out_idx is None
    # Exercise the real public output-attribute resolution without loading a
    # Hopper module on an A100 or requiring any GPU for this adapter regression.
    monkeypatch.setattr(importlib.import_module("tilelang.jit"), "cached", lambda **kwargs: kwargs)
    args = tilelang.compile(func, target=TARGETS["hopper"], out_idx=case.out_idx, pass_configs=case.pass_configs)
    assert args["out_idx"] == inferred and args["func"].same_as(func)


@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
def test_native_fp8_support_and_quantized_correctness(dtype):
    import torch
    from experiments.gemm_fp8.kernel import make_case

    w = replace(cases()[0], dtype=dtype)
    assert "FP8" in support_reason(w, Device("ampere", TARGETS["ampere"]))
    assert support_reason(w, Device("hopper", TARGETS["hopper"])) is None
    case = make_case(w)
    ref = torch.ones((16, 16)).to(getattr(torch, dtype))
    actual = ref.float()
    actual[0, 0] += torch.finfo(ref.dtype).eps
    case.check([actual.to(ref.dtype)], [ref])
    actual[0, 0] += torch.finfo(ref.dtype).eps
    with pytest.raises(AssertionError, match="quantization"):
        case.check([actual.to(ref.dtype)], [ref])
    # The spacing below 1 is half the spacing above 1. A symmetric tolerance
    # would incorrectly accept two representable steps on the lower side.
    actual = ref.float()
    actual[0, 0] -= torch.finfo(ref.dtype).eps / 2
    case.check([actual.to(ref.dtype)], [ref])
    actual[0, 0] -= torch.finfo(ref.dtype).eps / 2
    with pytest.raises(AssertionError, match="quantization"):
        case.check([actual.to(ref.dtype)], [ref])
    with pytest.raises(AssertionError):
        case.check([torch.zeros_like(ref)], [ref])
    with pytest.raises(AssertionError, match="dtype"):
        case.check([ref.float()], [ref])


@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
def test_fp8_example_cross_compiles_for_hopper(dtype):
    import tilelang
    from tvm.target import Target
    from experiments.gemm_fp8.kernel import make_case

    func = make_case(replace(cases()[0], dtype=dtype)).build(
        block_M=128, block_N=128, block_K=64, num_stages=2, threads=128, enable_rasteration=False
    )
    with Target(TARGETS["hopper"]) as target:
        artifact = tilelang.lower(func, target=target, enable_device_compile=True)
    assert "tl::wgmma_ss" in artifact.kernel_source
    assert "__nv_cvt_float2_to_fp8x2" in artifact.kernel_source
    instruction_dtype = "kFloat8_e4m3" if dtype == "float8_e4m3fn" else "kFloat8_e5m2"
    assert instruction_dtype in artifact.kernel_source


@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
def test_fp8_conversion_probe_cross_compiles(dtype):
    import tilelang
    from tvm.target import Target
    from tilelang.tiletune.profiling.device_probes import fp8_conversion

    with Target(TARGETS["hopper"]) as target:
        artifact = tilelang.lower(fp8_conversion(dtype, 3, 1), target=target, enable_device_compile=True)
    assert "cvt.rn.satfinite" in artifact.kernel_source


@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
def test_conversion_work_needs_a_dtype_specific_measured_rate(dtype):
    from experiments.gemm_fp8.kernel import make_case
    from tilelang.tiletune.src.collector import _Collector
    from tilelang.tiletune.compute import operation_work
    from tiletune_core.compute import estimate_phase_cycles

    func = make_case(replace(cases()[0], dtype=dtype)).build(
        block_M=128, block_N=128, block_K=64, num_stages=2, threads=128, enable_rasteration=False
    )
    work = operation_work(_Collector(func).operations[-1])
    name = f"convert_float32_to_{dtype}"
    assert work[name] == 128 * 128 and work["elementwise_ops"] == 0
    phase = dict(work=work, consumer_threads=128)
    assert estimate_phase_cycles(phase, dict(elementwise_ops_per_cycle=128), 1) is None
    rates = {name + "_per_cycle": 64, "consumer_rates": {"128": {name + "_per_cycle": 16}}}
    assert estimate_phase_cycles(phase, rates, 1) == 1024
    assert estimate_phase_cycles(phase, rates, 8) == 2048


def test_fp8_profile_merges_typed_and_common_consumer_rates(tmp_path):
    import json
    from tilelang.tiletune.profiling.device_profile import load_device_profile, _signature

    dtype = "float8_e4m3fn"
    name = f"convert_float32_to_{dtype}_per_cycle"
    signature = _signature(dtype, "float32")
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            dict(
                identity=dict(profile_version=7, target_arch="sm_90a"),
                common=dict(
                    rates=dict(elementwise_ops_per_cycle=64), clock_mhz=1000, consumer_rates={"128": dict(elementwise_ops_per_cycle=32)}
                ),
                gemm_models={json.dumps(signature, sort_keys=True): dict(rates={name: 64}, consumer_rates={"128": {name: 16}})},
            )
        )
    )
    profile = load_device_profile(path, input_dtype=dtype)
    assert profile["consumer_rates"]["128"] == {name: 16, "elementwise_ops_per_cycle": 32}
