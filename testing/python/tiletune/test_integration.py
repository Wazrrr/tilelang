"""Real CUDA coverage for exhaustive single/grouped compilation and JIT binding."""

import json
import pytest
import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import AutoTuner


def kernel(block=32, size=64, stages=0):
    @T.prim_func
    def main(A: T.Tensor((size, 64), "float16"), B: T.Tensor((64, size), "float16"), C: T.Tensor((size, size), "float32")):
        with T.Kernel(T.ceildiv(size, block), T.ceildiv(size, block), threads=128) as (bx, by):
            a = T.alloc_shared((block, 32), "float16")
            b = T.alloc_shared((32, block), "float16")
            c = T.alloc_fragment((block, block), "float32")
            T.clear(c)
            for k in T.Pipelined(2, num_stages=stages):
                T.copy(A[bx * block, k * 32], a)
                T.copy(B[k * 32, by * block], b)
                T.gemm(a, b, c)
            T.copy(c, C[bx * block, by * block])

    return main


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("mode", [None, "report_only", "reject"])
def test_gpu_exhaustive(tmp_path, grouped, mode):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    counts = []
    tuner = (
        AutoTuner(kernel, [{"block": 32}, {"block": 64}])
        .set_compile_args(target="cuda", execution_backend="tvm_ffi", out_idx=[2])
        .set_profile_args(ref_prog=lambda a, b: a.float() @ b.float(), rtol=0.01, atol=0.01)
    )
    if mode:
        tuner.set_tiletune_args(True, mode=mode, report_path=str(tmp_path / "report.json"))

        def elaborate(**config):
            config.pop("__pass_configs__", None)
            counts.append(config["block"])
            return kernel(**config)

        tuner.jit_elaborate = elaborate
    result = tuner.run(warmup=1, rep=2, enable_grouped_compile=grouped, group_compile_size=2)
    assert result.kernel is not None
    if mode:
        assert sorted(counts) == [32, 64]
        report = json.loads((tmp_path / "report.json").read_text())
        assert [r["status"] for r in report["configs"]] == ["benchmarked", "benchmarked"]
        assert all(r["pre_lowering"]["keep"] for r in report["configs"])
        assert all(r["compiler_resources"] for r in report["configs"])
        assert all("analysis_error" not in r for r in report["configs"])
        assert all(r["pressure"]["budget"] == 255 for r in report["configs"])
        assert all(r["pressure"]["budget_source"] == "architecture register limit" for r in report["configs"])
        assert all(r["post_compile"]["budget"] == 255 for r in report["configs"])
        assert report["settings"]["register_cap"] is None
        assert "analysis" in report["stage_cost_percent"]
        assert len(report["ranking"]) == 2
        assert sorted(r["index"] for r in report["ranking"]) == [0, 1]
        assert all(r["tile_cost"]["score"] is not None for r in report["configs"])
        assert all(r["tile_cost"]["input_bytes_per_block"] > 0 for r in report["configs"])


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("mode", ["reject", "report_only"])
def test_architecture_budget_controls_lowering(monkeypatch, tmp_path, grouped, mode):
    from tilelang.tiletune.runtime import TileTuneSession
    from tilelang.autotuner import grouped_compile as gc
    from tilelang.autotuner.param import CompileArgs
    from tvm.target import Target

    configs = [{"block": 128, "size": 256}, {"block": 256, "size": 256}]
    session = TileTuneSession(tilelang.TileTuneConfig(enabled=True, mode=mode, report_path=str(tmp_path / "report.json")), configs)
    reached = []

    def lower(program, **kwargs):
        reached.append(str(program.attrs["global_symbol"]))
        raise RuntimeError("test stops after reaching lowering")

    monkeypatch.setattr(gc, "lower_to_host_device_ir", lower)
    items = list(enumerate(configs))
    results = []
    for unit in [items] if grouped else [[item] for item in items]:
        results.extend(
            gc.compile_grouped_unit_tvm_ffi(
                unit, CompileArgs(target=Target({"kind": "cuda", "arch": "sm_90a"})), kernel, tiletune_session=session
            )
        )
    assert len(results) == 2
    assert len(reached) == (2 if mode == "report_only" else 1)
    assert [r["pressure"]["modeled_lower_bound"] for r in session.records] == [128, 512]
    assert all(r["pressure"]["budget"] == 255 for r in session.records)
    assert not session.records[0]["pre_lowering"]["would_reject"]
    assert session.records[1]["pre_lowering"]["would_reject"]
    if mode == "reject":
        assert session.records[1]["status"] == "pre_lowering_rejected"
    report = session.finish()
    assert report["settings"]["register_cap"] is None
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["configs"][1]["pressure"]["budget_source"] == "architecture register limit"


@pytest.mark.parametrize("grouped", [False, True])
def test_rejection_never_lowers_and_failure_attribution(monkeypatch, tmp_path, grouped):
    from tilelang.tiletune.runtime import TileTuneSession
    from tilelang.autotuner import grouped_compile as gc
    from tilelang.autotuner.param import CompileArgs
    from tvm.target import Target

    session = TileTuneSession(tilelang.tiletune.TileTuneConfig(enabled=True, register_cap=1), [{"block": 32}, {"block": -1}])
    calls = []

    def elaborate(block):
        calls.append(block)
        if block == -1:
            raise ValueError("elaboration failed")
        return kernel(block)

    monkeypatch.setattr(gc, "lower_to_host_device_ir", lambda *a, **k: pytest.fail("rejected config reached lowering"))
    items = [(0, {"block": 32}), (1, {"block": -1})]
    units = [items] if grouped else [[item] for item in items]
    results = []
    for unit in units:
        results.extend(gc.compile_grouped_unit_tvm_ffi(unit, CompileArgs(target=Target("cuda")), elaborate, tiletune_session=session))
    assert len(results) == 2
    assert calls == [32, -1]
    assert [r["status"] for r in session.records] == ["pre_lowering_rejected", "elaboration_failed"]
    assert session.records[0]["pressure"]["modeled_lower_bound"] == 8


def test_gpu_mma_wgmma_and_function_settings(tmp_path):
    from tilelang.autotuner.grouped_compile import compile_grouped_unit_tvm_ffi
    from tilelang.autotuner.param import CompileArgs
    from tilelang.tiletune.runtime import TileTuneSession
    from tvm.target import Target

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Hopper required")
    counts = []

    def elaborate(disable_wgmma):
        counts.append(disable_wgmma)
        return kernel(64).with_attr("tilelang_pass_configs", {"tl.disable_wgmma": disable_wgmma})

    for mode in ("report_only", "reject"):
        configs = [{"disable_wgmma": True}, {"disable_wgmma": False}]
        session = TileTuneSession(tilelang.TileTuneConfig(enabled=True, mode=mode), configs)
        results = compile_grouped_unit_tvm_ffi(
            list(enumerate(configs)),
            CompileArgs(target=Target({"kind": "cuda", "arch": "sm_90a"}), out_idx=[2], execution_backend="tvm_ffi"),
            elaborate,
            tiletune_session=session,
        )
        assert len(results) == 2
        for idx, _, jit, error in results:
            assert error is None, str(error)
            source = jit.get_kernel_source()
            assert ("tl::wgmma" in source) == (idx == 1)
            a = torch.randn((64, 64), device="cuda", dtype=torch.float16)
            b = torch.randn_like(a)
            torch.testing.assert_close(jit(a, b), a.float() @ b.float(), rtol=0.01, atol=0.01)
            assert session.records[idx]["effective_pass_configs"]["tl.disable_wgmma"] == (idx == 0)
        assert all(r["post_compile"]["keep"] for r in session.records)
    assert counts == [True, False, True, False]


def test_decorator_binding_and_fresh_report(monkeypatch, tmp_path):
    tuned = tilelang.autotune(
        configs=[{"block": 32, "pass_configs": {"tl.disable_wgmma": True}}, {"block": 64}],
        tiletune={"mode": "report_only", "report_path": str(tmp_path / "fresh.json")},
        warmup=1,
        rep=1,
        ref_prog=lambda a, b: a.float() @ b.float(),
    )(tilelang.jit(out_idx=[2], execution_backend="tvm_ffi")(kernel))
    calls = []
    original = tuned.jit_impl.get_tir

    def get_tir(*args, **kwargs):
        calls.append(kwargs["block"])
        return original(*args, **kwargs)

    monkeypatch.setattr(tuned.jit_impl, "get_tir", get_tir)
    assert tuned() is not None
    assert tuned() is not None
    assert sorted(calls) == [32, 32, 64, 64]
    assert all(r["status"] == "benchmarked" for r in tuned.tiletune_report["configs"])
    assert all(r["pressure"]["budget"] == 255 for r in tuned.tiletune_report["configs"])
    assert tuned.tiletune_report["configs"][0]["effective_pass_configs"]["tl.disable_wgmma"]


def test_analysis_failure_stops_before_lowering(monkeypatch):
    from tilelang.tiletune.runtime import TileTuneSession
    from tilelang.autotuner import grouped_compile as gc
    from tilelang.autotuner.param import CompileArgs
    from tvm.target import Target

    session = TileTuneSession(tilelang.TileTuneConfig(enabled=True), [{"block": 32}, {"block": 64}])
    reached = []

    def unknown(*args, **kwargs):
        raise RuntimeError("unsupported analysis")

    def lower(program, **kwargs):
        reached.append(str(program.attrs["global_symbol"]))
        raise RuntimeError("test stops after reaching lowering")

    monkeypatch.setattr("tilelang.tiletune.runtime.analyze_prim_func", unknown)
    monkeypatch.setattr(gc, "lower_to_host_device_ir", lower)
    results = gc.compile_grouped_unit_tvm_ffi(
        [(0, {"block": 32}), (1, {"block": 64})], CompileArgs(target=Target("cuda")), kernel, tiletune_session=session
    )
    assert len(results) == 2
    assert not reached
    assert all(error is not None for _, _, _, error in results)
    assert all(r["status"] == "analysis_failed" and r["pre_lowering"] is None for r in session.records)
    assert all(r["analysis_error"] == "unsupported analysis" for r in session.records)


@pytest.mark.parametrize("grouped", [False, True])
def test_gpu_ws_policy_matches_compiled_launch_and_function_settings(grouped):
    import re
    from tilelang.autotuner.grouped_compile import compile_grouped_unit_tvm_ffi
    from tilelang.autotuner.param import CompileArgs
    from tilelang.tiletune.runtime import TileTuneSession
    from tvm.target import Target

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Hopper required")
    configs = [{"stages": 0, "disable": False}, {"stages": 2, "disable": False}, {"stages": 2, "disable": True}]
    session = TileTuneSession(tilelang.TileTuneConfig(enabled=True, mode="report_only"), configs)
    calls = []

    def elaborate(stages, disable):
        calls.append((stages, disable))
        return kernel(64, stages=stages).with_attr("tilelang_pass_configs", {"tl.disable_warp_specialized": disable})

    items = list(enumerate(configs))
    results = []
    for unit in [items] if grouped else [[item] for item in items]:
        results.extend(
            compile_grouped_unit_tvm_ffi(
                unit,
                CompileArgs(target=Target({"kind": "cuda", "arch": "sm_90a"}), out_idx=[2], execution_backend="tvm_ffi"),
                elaborate,
                tiletune_session=session,
            )
        )
    assert len(calls) == 3
    for idx, _, jit, error in results:
        assert error is None, str(error)
        policy = session.records[idx]["pressure"]["warp_specialization"]
        assert policy["status"] == ["not_applicable", "predicted", "disabled"][idx]
        src = jit.get_kernel_source()
        name = next(iter(session.records[idx]["compiler_resources"]))
        # Grouped JIT kernels share a source module; inspect this entry only.
        entry = re.search(r"__launch_bounds__\((\d+),[^)]*\)\s+" + re.escape(name) + r"\([^\n]*\{[\s\S]*?\n\}", src)
        assert entry is not None
        launch = int(entry[1])
        assert launch == (256 if idx == 1 else 128)
        assert ("warpgroup_reg_alloc<240>" in entry[0]) == (idx == 1)
        a = torch.randn((64, 64), device="cuda", dtype=torch.float16)
        b = torch.randn_like(a)
        torch.testing.assert_close(jit(a, b), a.float() @ b.float(), rtol=0.01, atol=0.01)


def test_ws_analysis_receives_effective_compile_override(monkeypatch):
    from tilelang.autotuner import grouped_compile as gc
    from tilelang.autotuner.param import CompileArgs
    from tilelang.tiletune.runtime import TileTuneSession
    from tvm.target import Target

    session = TileTuneSession(tilelang.TileTuneConfig(enabled=True), [{}])

    def elaborate():
        return kernel(64, stages=2).with_attr("tilelang_pass_configs", {"tl.disable_warp_specialized": True})

    def lower(*args, **kwargs):
        raise RuntimeError("stop at lowering")

    monkeypatch.setattr(gc, "lower_to_host_device_ir", lower)
    gc.compile_grouped_unit_tvm_ffi(
        [(0, {})],
        CompileArgs(target=Target({"kind": "cuda", "arch": "sm_90a"}), pass_configs={"tl.disable_warp_specialized": False}),
        elaborate,
        tiletune_session=session,
    )
    assert session.records[0]["pressure"]["warp_specialization"]["status"] == "predicted"
    assert not session.records[0]["effective_pass_configs"]["tl.disable_warp_specialized"]
