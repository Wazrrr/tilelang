"""Shared grouped-build failures must not discard valid neighboring configs."""

from collections import Counter
import re
from types import SimpleNamespace

import pytest

import tilelang.language as T
from tilelang import tvm
from tilelang.autotuner import AutoTuner, grouped_compile as gc
from tilelang.autotuner.filters.launch import LaunchResourceInfo
from tilelang.autotuner.param import CompileArgs
from tilelang.contrib.cuda_resource_info import parse_ptxas_output, record_usage
from tilelang.tiletune import TileTuneConfig, TileTuneReject
from tilelang.tiletune.runtime import TileTuneSession


def _indices(mod):
    indices = set()
    for gv in mod.functions:
        match = re.search(r"_gc_(\d+)", str(gv.name_hint))
        if match is not None:
            indices.add(int(match[1]))
    return sorted(indices)


@pytest.fixture
def backend(monkeypatch):
    """Inject failures while exercising the actual grouping and spill policy."""
    state = SimpleNamespace(
        elaborated=[],
        lowered=[],
        device_calls=[],
        host_calls=[],
        bad_device=set(),
        bad_host=set(),
        bad_lower=set(),
        bad_elaborate=set(),
        spills=set(),
    )

    def elaborate(variant):
        state.elaborated.append(variant)
        if variant in state.bad_elaborate:
            raise ValueError(f"invalid elaboration {variant}")
        return tvm.tirx.PrimFunc([], tvm.tirx.Evaluate(0)).with_attr("global_symbol", "kernel")

    def lower(program, *, target, target_host):
        symbol = str(program.attrs["global_symbol"])
        idx = int(symbol.rsplit("_gc_", 1)[1])
        state.lowered.append(idx)
        if idx in state.bad_lower:
            raise ValueError(f"invalid lowering {idx}")
        mod = tvm.IRModule({symbol: program})
        return mod, mod, [], target, target_host

    def device_codegen(mod, target):
        group = _indices(mod)
        state.device_calls.append(group)
        bad = sorted(set(group) & state.bad_device)
        if bad:
            raise RuntimeError(f"invalid device configs {bad}")
        for gv in mod.functions:
            name = str(gv.name_hint)
            idx = int(name.rsplit("_gc_", 1)[1])
            spill = 4 if idx in state.spills else 0
            record_usage(
                parse_ptxas_output(
                    f"""ptxas info : Function properties for {name}
    0 bytes stack frame, {spill} bytes spill stores, 0 bytes spill loads
ptxas info : Used 16 registers
"""
                )
            )
        return SimpleNamespace(inspect_source=lambda: "source")

    def host_codegen(mod, target_host, *, target):
        group = _indices(mod)
        state.host_calls.append(group)
        bad = sorted(set(group) & state.bad_host)
        if bad:
            raise RuntimeError(f"invalid host configs {bad}")
        return SimpleNamespace(import_module=lambda module: None)

    monkeypatch.setattr(gc, "lower_to_host_device_ir", lower)
    monkeypatch.setattr(gc, "device_codegen", device_codegen)
    monkeypatch.setattr(gc, "host_codegen", host_codegen)
    monkeypatch.setattr(
        gc,
        "extract_launch_resource_info",
        lambda mod: [LaunchResourceInfo(function_name=str(gv.name_hint), block_dims=(32, 1, 1)) for gv in mod.functions],
    )
    monkeypatch.setattr(gc.tvm.runtime, "Executable", lambda mod: SimpleNamespace(jit=lambda: None))
    monkeypatch.setattr(gc, "TVMFFIKernelAdapter", lambda **kwargs: SimpleNamespace(func=lambda: None))
    monkeypatch.setattr(gc, "JITKernel", lambda **kwargs: SimpleNamespace())
    state.elaborate = elaborate
    return state


def _compile_eight(backend, *, session=None):
    configs = [dict(variant=i) for i in range(8)]
    results = gc.compile_grouped_unit_tvm_ffi(
        list(enumerate(configs)),
        CompileArgs(target=tvm.target.Target({"kind": "cuda", "arch": "sm_100a"}), execution_backend="tvm_ffi"),
        backend.elaborate,
        tiletune_session=session,
    )
    assert len(results) == 8
    assert Counter(idx for idx, _, _, _ in results) == Counter(range(8))
    assert backend.elaborated == list(range(8))
    return {idx: (kernel, error) for idx, _, kernel, error in results}


@pytest.mark.parametrize("stage", ["device", "host"])
@pytest.mark.parametrize("bad", [{3}, {1, 6}, set(range(8))])
def test_shared_failure_isolated_to_invalid_configs(backend, stage, bad):
    setattr(backend, f"bad_{stage}", bad)
    results = _compile_eight(backend)
    for idx, (kernel, error) in results.items():
        assert (kernel is None) == (idx in bad)
        assert (error is not None) == (idx in bad)
    # Binary splitting takes at most 15 shared/singleton attempts for 8 configs.
    assert len(backend.device_calls) <= 15
    assert len(backend.host_calls) <= 15


def test_all_valid_configs_keep_one_shared_build(backend):
    results = _compile_eight(backend)
    assert all(kernel is not None and error is None for kernel, error in results.values())
    assert backend.device_calls == backend.host_calls == [list(range(8))]
    assert all(kernel.adapter._autotune_group_size == 8 for kernel, _ in results.values())


def test_fallback_preserves_per_config_failures_and_spill_rejections(backend, monkeypatch):
    backend.bad_elaborate = {0}
    backend.bad_lower = {1}
    backend.spills = {2, 5}
    backend.bad_host = {3}
    configs = [dict(variant=i) for i in range(8)]
    session = TileTuneSession(
        TileTuneConfig(enabled=True, mode="reject", max_spill_bytes=0, max_local_bytes=0),
        configs,
        target={"kind": "cuda", "arch": "sm_100a"},
    )
    # Selection is already frozen; this test exercises compilation recovery and
    # the real post-compile resource decision without rerunning IR analysis.
    monkeypatch.setattr(session, "elaborate", lambda idx, config, elaborate, **kwargs: elaborate(**config))
    session.selection = dict(selected_indices=list(range(8)), selected_count=8)
    results = _compile_eight(backend, session=session)
    assert isinstance(results[0][1], ValueError)
    assert isinstance(results[1][1], ValueError)
    assert str(results[3][1]) == "invalid host configs [3]"
    for idx in backend.spills:
        assert isinstance(results[idx][1], TileTuneReject)
        assert session.records[idx]["status"] == "post_compile_rejected"
        assert not session.records[idx].get("grouped_compile_fallbacks")
        assert all(idx not in group for group in backend.device_calls[1:])
    assert all(results[idx][0] is not None for idx in {4, 6, 7})
    assert session.selection == dict(selected_indices=list(range(8)), selected_count=8)
    assert all(session.records[idx]["grouped_compile_fallbacks"] for idx in {3, 4, 6, 7})


@pytest.mark.parametrize("stage", ["device", "host"])
def test_compile_cost_includes_failed_shared_attempts(backend, monkeypatch, stage):
    setattr(backend, f"bad_{stage}", {3})
    clock = [0.0]
    monkeypatch.setattr(gc.time, "perf_counter", lambda: clock[0])
    device_compile, host_compile = gc.device_codegen, gc.host_codegen

    def device(*args, **kwargs):
        clock[0] += 1
        return device_compile(*args, **kwargs)

    def host(*args, **kwargs):
        clock[0] += 2
        return host_compile(*args, **kwargs)

    monkeypatch.setattr(gc, "device_codegen", device)
    monkeypatch.setattr(gc, "host_codegen", host)
    session = TileTuneSession(
        TileTuneConfig(enabled=True, mode="report_only", max_spill_bytes=0, max_local_bytes=0),
        [dict(variant=i) for i in range(8)],
        target={"kind": "cuda", "arch": "sm_100a"},
    )
    monkeypatch.setattr(session, "elaborate", lambda idx, config, elaborate, **kwargs: elaborate(**config))
    _compile_eight(backend, session=session)
    costs = session.finish()["stage_cost_ms"]
    assert costs["device_compile"] == 1000 * len(backend.device_calls)
    assert costs["host_compile"] == 2000 * len(backend.host_calls)


def _copy_kernel(block=32, threads=32):
    @T.prim_func
    def main(A: T.Tensor((256,), "float32"), B: T.Tensor((256,), "float32")):
        with T.Kernel(T.ceildiv(256, block), threads=threads) as bx:
            for i in T.Parallel(block):
                B[bx * block + i] = A[bx * block + i] + 1

    return main


@pytest.mark.parametrize("inject_failure", [False, True])
def test_gpu_pipeline_alpha_group_eight(monkeypatch, tmp_path, inject_failure):
    """Exercise fallback through the real B200 lowering/codegen/runtime path."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")
    monkeypatch.setenv("TILELANG_AUTO_TUNING_DISABLE_CACHE", "1")
    monkeypatch.setenv("TILELANG_AUTO_TUNING_CPU_COUNTS", "2")
    original_codegen = gc.device_codegen
    original_lower = gc.lower_to_host_device_ir
    lowered = []

    def codegen(mod, target):
        if inject_failure and 3 in _indices(mod):
            raise RuntimeError("injected device compiler failure for config 3")
        return original_codegen(mod, target)

    def lower(program, **kwargs):
        lowered.append(str(program.attrs["global_symbol"]))
        return original_lower(program, **kwargs)

    monkeypatch.setattr(gc, "device_codegen", codegen)
    monkeypatch.setattr(gc, "lower_to_host_device_ir", lower)
    configs = [dict(block=block, threads=threads) for block in (32, 64) for threads in range(32, 257, 32)]
    tuner = (
        AutoTuner(_copy_kernel, configs)
        .set_compile_args(target="cuda", execution_backend="tvm_ffi", out_idx=[1])
        .set_profile_args(ref_prog=lambda a: a + 1, backend="event")
        .set_tiletune_args(
            TileTuneConfig(
                enabled=True,
                mode="reject",
                ranking_metric="memory",
                alpha=0.5,
                max_spill_bytes=0,
                max_local_bytes=0,
                report_path=str(tmp_path / "report.json"),
            )
        )
    )
    winner = tuner.run(warmup=1, rep=2, use_pipeline=True, enable_grouped_compile=True, group_compile_size=8)
    report = tuner.tiletune_report
    assert report["selection"]["selected_indices"] == list(range(8))
    assert all(record["status"] == "not_selected" for record in report["configs"][8:])
    for idx, record in enumerate(report["configs"][:8]):
        assert record["status"] == ("compilation_failed" if inject_failure and idx == 3 else "benchmarked")
        if not (inject_failure and idx == 3):
            assert record["post_compile"]["keep"]
            assert record["compiler_resources"]
    assert len(lowered) >= 8
    value = torch.randn(256, device="cuda")
    torch.testing.assert_close(winner.kernel(value), value + 1)
