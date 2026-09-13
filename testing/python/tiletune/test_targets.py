"""Hardware identity, model boundaries and heterogeneous resource observations."""

import json
from types import SimpleNamespace

import pytest

from tilelang.tiletune import TileTuneConfig, analyze_prim_func, check_compiler_resources, current_target, resolve_target
from tilelang.tiletune.compute import fragment_reduction_work
from tilelang.tiletune.profiling.device_profile import load_device_profile
from tilelang.tiletune.src.device import query_device_limits
from test_pipeline import matrix_pipeline
from test_modules import PROFILE
from examples.gemm.example_gemm_tiletune_trace import ILLUSTRATIVE_LIMITS


@pytest.mark.parametrize(
    "target,family,warp,cap",
    [
        ({"kind": "cuda", "arch": "sm_80"}, "ampere", 32, 255),
        ("cuda -arch=sm_90a", "hopper", 32, 255),
        ({"kind": "cuda", "arch": "sm_100a"}, "blackwell", 32, 255),
        ({"kind": "cuda", "arch": "sm_120"}, "blackwell", 32, 255),
        ({"kind": "hip", "mcpu": "gfx942:sramecc+:xnack-"}, "cdna3", 64, None),
        ({"kind": "ascendc", "arch": "Ascend910"}, "ascend", None, None),
    ],
)
def test_offline_target_identity(target, family, warp, cap, monkeypatch):
    import torch

    def forbidden(*args, **kwargs):
        raise AssertionError("offline target resolution must not query a device")

    monkeypatch.setattr(torch.cuda, "get_device_properties", forbidden)
    model = resolve_target(target)
    assert (model.architecture, model.subgroup_size, model.register_cap) == (family, warp, cap)


def test_hip_detection_and_queries_do_not_use_cuda_attributes(monkeypatch):
    import torch

    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(gcnArchName="gfx942:sramecc+", multi_processor_count=4, warp_size=64),
    )
    assert current_target()["mcpu"] == "gfx942"
    assert query_device_limits({"kind": "hip", "mcpu": "gfx942"}) == {"sm_count": 4, "warp_size": 64}
    assert query_device_limits({"kind": "cuda", "arch": "sm_90a"}) is None


def test_hip_register_units_and_symbolic_metadata():
    from tilelang.contrib.hip_resource_info import KernelResourceUsage

    target = {"kind": "hip", "mcpu": "gfx942"}
    usage = KernelResourceUsage(extra={"VGPRs": "kernel.num_vgpr", "AGPRs": "0", "ScratchSize [bytes/lane]": "kernel.private_seg_size"})
    result = check_compiler_resources({"f": usage}, ["f"], target=target)
    assert result["keep"] and result["status"] == "unknown"
    assert result["resources"]["f"]["registers"] is None
    assert result["resources"]["f"]["local_bytes"] is None
    usage.extra = {"VGPRs": "20", "AGPRs": "12", "TotalSGPRs": "8", "ScratchSize [bytes/lane]": "0"}
    result = check_compiler_resources({"f": usage}, ["f"], TileTuneConfig(register_cap=31), target=target)
    counters = result["resources"]["f"]
    assert counters["registers"] == 32 and counters["scalar_registers"] == 8
    assert not result["keep"]
    usage.extra.pop("AGPRs")
    assert check_compiler_resources({"f": usage}, ["f"], target=target)["resources"]["f"]["registers"] is None


def test_subgroup_reduction_uses_64_lane_ownership():
    import tilelang.language as T

    layout = T.Fragment((64,), forward_fn=lambda i: (i, 0))
    result = fragment_reduction_work(layout, [64], 0, subgroup_size=64)
    assert result["lane_widths"] == [64]
    assert result["shuffle_pairs"] == 64 * 6
    with pytest.raises(ValueError, match="inter-warp"):
        fragment_reduction_work(layout, [64], 0, subgroup_size=32)


def test_hip_instruction_and_profile_do_not_inherit_cuda():
    target = {"kind": "hip", "mcpu": "gfx942", "thread_warp_size": 64}
    limits = dict(ILLUSTRATIVE_LIMITS, warp_size=64)
    wrong = dict(PROFILE, profile_backend="cuda", profile_target="sm_90a")
    result = analyze_prim_func(
        matrix_pipeline(stages=0), dict(ranking_metric="pipeline_time", performance_model=wrong), target=target, device_limits=limits
    )
    phases = result["modules"]["pipeline_overlap"]["phases"]
    for phase in phases:
        if phase["work"]["gemm_flops"]:
            compute = phase["compute_participants"]
            # A CUDA-only native build cannot query the HIP operator registry.
            # It must report unknown instead of substituting CUDA instructions.
            if compute["precision"] == "predicted":
                assert compute["instruction"] == "rocm.mfma"
            else:
                assert "no gemm implementation is registered" in compute["reason"]
    assert result["tile_cost"]["score"] is None
    assert result["pressure"]["hardware_register_cap"] is None
    assert result["pressure"]["warp_specialization"]["status"] == "not_applicable"


def test_ascend_keeps_logical_analysis_without_cuda_residency():
    result = analyze_prim_func(
        matrix_pipeline(stages=0), target={"kind": "ascendc", "arch": "Ascend910"}, device_limits=ILLUSTRATIVE_LIMITS
    )
    assert result["tile_propagation"]
    assert result["tile_cost"]["score"] is None
    assert "target requires a non-SIMT core/storage residency model" in result["modules"]["waves"]["unknown"]


def test_external_hip_profile_is_offline_and_instruction_specific(tmp_path):
    signature = dict(instruction="rocm.mfma", a_dtype="float16", b_dtype="float16", accum_dtype="float32")
    path = tmp_path / "mi308.json"
    path.write_text(
        json.dumps(
            dict(
                identity=dict(profile_version=4, backend="hip", target_arch="gfx942", matrix_instruction="rocm.mfma"),
                common=dict(rates=PROFILE, clock_mhz=1000),
                gemm_models={json.dumps(signature, sort_keys=True): dict(rates={})},
            )
        )
    )
    model = load_device_profile(path, input_dtype="float16")
    assert model["profile_backend"] == "hip"
    assert model["gemm_signature"] == signature


def test_blackwell_mma_probe_cross_compiles():
    import tilelang
    from tilelang.tiletune.profiling.device_profile import _instruction
    from tilelang.tiletune.profiling.device_probes import tensor_core

    assert _instruction(dict(target_arch="sm_100a")) == "cuda.mma"
    # Compiler validation only; this test makes no Blackwell performance claim.
    from tvm.target import Target

    with Target({"kind": "cuda", "arch": "sm_100a"}) as target:
        artifact = tilelang.lower(tensor_core("float16", "float32", 3, 1, 128), target=target, enable_device_compile=True)
    assert "tl::mma_sync" in artifact.kernel_source


def test_recurrent_kda_traffic_and_timing_boundary():
    from experiments.portable.spec import Device, TARGETS, configurations, default_workloads
    from experiments.portable.kernels import make_case

    workload = next(w for w in default_workloads(True) if w.name == "kda_recurrent")
    case = make_case(workload)
    func = case.build(**configurations(workload, Device("hopper", TARGETS["hopper"]))[0])
    report = analyze_prim_func(func, target=TARGETS["hopper"], device_limits=ILLUSTRATIVE_LIMITS)
    memory = report["modules"]["memory_traffic"]
    assert memory["traffic_bytes_per_block"] == 7232
    assert {t["buffer"] for t in memory["input_tiles"]} == {"Q", "K", "V", "G", "Beta"}
    assert report["tile_cost"]["score"] is not None
    assert (
        "direct scalar global accesses inside a recurrence require a per-iteration access schedule"
        in report["modules"]["pipeline_overlap"]["unknown"]
    )


def test_scalar_thread_access_does_not_report_one_value_per_cta():
    from tvm import tirx as tir
    from tvm.ir import Range

    a = tir.decl_buffer((128,), "float32", name="A")
    c = tir.decl_buffer((128,), "float32", name="C")
    x = tir.decl_buffer((1,), "float32", name="x", scope="local")
    tx, bx = tir.Var("tx", "int32"), tir.Var("bx", "int32")
    body = tir.SeqStmt([tir.BufferStore(x, a[tx] + 1, [0]), tir.BufferStore(c, x[0], [tx])])
    body = tir.SBlockRealize([], True, tir.SBlock([], [], [], "root", body, alloc_buffers=[x]))
    for var, extent, tag in [(tx, 128, "threadIdx.x"), (bx, 1, "blockIdx.x")]:
        body = tir.For(
            var,
            0,
            extent,
            tir.ForKind.THREAD_BINDING,
            body,
            thread_binding=tir.IterVar(Range(0, extent), var, tir.IterVar.ThreadIndex, tag),
        )
    func = tir.PrimFunc([a.data, c.data], body, buffer_map={a.data: a, c.data: c})
    report = analyze_prim_func(func, target={"kind": "cuda", "arch": "sm_80"}, device_limits=ILLUSTRATIVE_LIMITS)
    # The CTA reads and writes 128 floats, not one. Until lane coverage is
    # represented, retaining a thread domain only for clipping is unsound.
    assert report["tile_cost"]["score"] is None


def test_hip_compile_reuses_frozen_ir_and_preserves_flag_ownership(monkeypatch):
    import tilelang
    from tilelang.autotuner.param import CompileArgs
    from tilelang.tiletune.runtime import TileTuneSession
    from tilelang.contrib.hip_resource_info import KernelResourceUsage
    from experiments.portable.spec import Workload
    from experiments.portable.kernels import make_case

    target = {"kind": "hip", "mcpu": "gfx942", "thread_warp_size": 64}
    configs = [{"block_rows": 1, "threads": 128}, {"block_rows": 2, "threads": 128}]
    case = make_case(Workload("softmax", "softmax", dict(rows=32, columns=128)))
    built = []

    def elaborate(**kwargs):
        func = case.build(**kwargs).with_attr("tilelang_compile_flags", ["-DFUNCTION_FLAG=1"])
        built.append(func)
        return func

    session = TileTuneSession(
        TileTuneConfig(enabled=True, mode="report_only", top_k=1),
        configs,
        ["-DSESSION_FLAG=1"],
        target=target,
        device_limits=dict(ILLUSTRATIVE_LIMITS, warp_size=64),
    )
    selected = session.prepare_top_k([(i, c, {}) for i, c in enumerate(configs)], elaborate)
    assert len(selected) == 1
    idx = selected[0]
    frozen = session.prepared_programs[idx]

    def compile_program(program, **kwargs):
        assert program is frozen
        assert list(program.attrs["tilelang_compile_flags"]) == ["-DFUNCTION_FLAG=1"]
        assert kwargs["pass_configs"]["tl.device_compile_flags"] == ["-DSESSION_FLAG=1"]
        assert kwargs["target"] == target
        return SimpleNamespace(resource_usage={"kernel": KernelResourceUsage(extra={"VGPRs": "20", "AGPRs": "0"})})

    monkeypatch.setattr(tilelang, "compile", compile_program)
    session.compile_native(idx, configs[idx], elaborate, CompileArgs(target=target, execution_backend="tvm_ffi"))
    assert len(built) == len(configs)
    assert session.records[idx]["compiler_resources"]["kernel"]["registers"] == 20
