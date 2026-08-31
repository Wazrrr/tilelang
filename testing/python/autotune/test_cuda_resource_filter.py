from __future__ import annotations

import importlib
import inspect

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.autotuner import AutoTuner
from tilelang.autotuner.filters import (
    CudaDeviceLimits,
    LaunchResourceInfo,
    evaluate_post_compile_resource_filter,
    evaluate_pre_compile_resource_filter,
    extract_launch_resource_info,
)
from tilelang.cache.cuda_binary_cache import CUDABinaryCache
from tilelang.cache.kernel_cache import KernelCache
from tilelang.contrib.cuda_resource_info import KernelResourceUsage
from tilelang.contrib.cuda_resource_info import CUDA_RESOURCE_CAPTURE_CONFIG_KEY
from tilelang.contrib.cuda_resource_info import pop_recorded as cuda_pop_recorded
from tilelang.contrib.cuda_resource_info import reset_recorder as cuda_reset_recorder
from tilelang.env import env
from tvm.target import Target


def _limits() -> CudaDeviceLimits:
    return CudaDeviceLimits(
        max_threads_per_block=1024,
        max_block_dims=(1024, 1024, 64),
        max_grid_dims=(2**31 - 1, 65535, 65535),
        max_shared_memory_per_block=49152,
        max_shared_memory_per_block_optin=228 * 1024,
        max_registers_per_block=65536,
        cooperative_launch=True,
    )


def _set_cache_dirs(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    tmp_dir = tmp_path / "tmp"
    cache_dir.mkdir()
    tmp_dir.mkdir()
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(env, "TILELANG_TMP_DIR", str(tmp_dir))
    monkeypatch.setattr(env, "TILELANG_DISABLE_CACHE", "0")
    tilelang.enable_cache()
    AutoTuner._memory_cache.clear()
    KernelCache._get_cache_namespace.cache_clear()
    CUDABinaryCache._get_tilelang_lib_stamp.cache_clear()


def _make_small_matmul_kernel():
    def kernel(block_M=None, block_N=None, block_K=None):
        @T.prim_func
        def main(
            A: T.Tensor((128, 128), T.float16),
            B: T.Tensor((128, 128), T.float16),
            C: T.Tensor((128, 128), T.float16),
        ):
            with T.Kernel(T.ceildiv(128, block_N), T.ceildiv(128, block_M), threads=128) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), T.float16)
                B_shared = T.alloc_shared((block_N, block_K), T.float16)
                C_local = T.alloc_fragment((block_M, block_N), T.float32)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(128, block_K), num_stages=1):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                T.copy(C_local, C[by * block_M, bx * block_N])

        return main

    return kernel


def test_extract_launch_resource_info_from_device_ir_attrs():
    func = (
        tvm.tirx.PrimFunc([], tvm.tirx.Evaluate(0))
        .with_attr("global_symbol", "main_kernel")
        .with_attr(
            "thread_extent",
            {
                "threadIdx.x": tvm.tirx.IntImm("int32", 256),
                "threadIdx.y": tvm.tirx.IntImm("int32", 2),
                "blockIdx.x": tvm.tirx.IntImm("int32", 80),
            },
        )
        .with_attr("dyn_shared_memory_buf", tvm.tirx.IntImm("int64", 16384))
    )
    mod = tvm.IRModule({"main_kernel": func})

    [info] = extract_launch_resource_info(mod)

    assert info.function_name == "main_kernel"
    assert info.block_dims == (256, 2, 1)
    assert info.grid_dims == (80, 1, 1)
    assert info.threads_per_block == 512
    assert info.dynamic_smem_bytes == 16384


def test_pre_compile_filter_rejects_dynamic_smem_over_optin_limit():
    decision = evaluate_pre_compile_resource_filter(
        [LaunchResourceInfo("main_kernel", block_dims=(256, 1, 1), dynamic_smem_bytes=300 * 1024)],
        _limits(),
    )

    assert decision.verdict == "reject"
    assert decision.reason == "dynamic_shared_memory_over_limit"


def test_pre_compile_filter_keeps_unknown_values():
    decision = evaluate_pre_compile_resource_filter(
        [LaunchResourceInfo("main_kernel", block_dims=(None, 1, 1), dynamic_smem_bytes=None)],
        _limits(),
    )

    assert decision.verdict == "keep"


def test_post_compile_filter_rejects_register_overbooking():
    decision = evaluate_post_compile_resource_filter(
        [LaunchResourceInfo("main_kernel", block_dims=(512, 1, 1), dynamic_smem_bytes=0)],
        {"main_kernel": KernelResourceUsage(n_regs=256)},
        _limits(),
    )

    assert decision.verdict == "reject"
    assert decision.reason == "registers_per_block_over_limit"


def test_cuda_compile_callback_records_and_replays_cached_resource_usage(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)
    lower = importlib.import_module("tilelang.engine.lower")
    monkeypatch.setattr(env, "TILELANG_KERNEL_CACHE_USE_LIB_STAMP", "0")

    compile_calls = []
    output = """
ptxas info    : Compiling entry function 'main_kernel' for 'sm_90'
ptxas info    : Used 64 registers, 2048 bytes smem, 16 bytes cmem[0]
"""

    def fake_compile_cuda(code, target_format, arch, options=None, verbose=False, return_output=False):
        compile_calls.append((code, target_format, tuple(arch), tuple(options or ()), return_output))
        assert return_output is True
        return bytearray(b"fake-cubin"), output

    monkeypatch.setattr(lower.nvcc, "compile_cuda", fake_compile_cuda)

    target = Target({"kind": "cuda", "arch": "sm_90"})
    source = 'extern "C" __global__ void main_kernel() {}'
    pass_configs = {CUDA_RESOURCE_CAPTURE_CONFIG_KEY: True}

    cuda_reset_recorder()
    first = lower.tilelang_callback_cuda_compile(source, target, pass_configs)
    first_usage = cuda_pop_recorded()

    cuda_reset_recorder()
    second = lower.tilelang_callback_cuda_compile(source, target, pass_configs)
    second_usage = cuda_pop_recorded()

    assert bytes(first) == b"fake-cubin"
    assert bytes(second) == b"fake-cubin"
    assert len(compile_calls) == 1
    assert first_usage["main_kernel"].n_regs == 64
    assert second_usage["main_kernel"].n_regs == 64


def test_autotuner_resource_filter_args_are_deprecated_noop(tmp_path):
    report_path = tmp_path / "resource_filter_report.tsv"
    tuner = AutoTuner.from_kernel(
        kernel=_make_small_matmul_kernel(),
        configs=[{"block_M": 64, "block_N": 64, "block_K": 32}],
    ).set_resource_filter_args(True, report_path=str(report_path))

    assert tuner.resource_filter_args.enabled is True
    assert tuner.resource_filter_args.report_path == str(report_path)


def test_autotuner_resource_filter_args_do_not_affect_cache_key(tmp_path):
    tuner = AutoTuner.from_kernel(
        kernel=_make_small_matmul_kernel(),
        configs=[{"block_M": 64, "block_N": 64, "block_K": 32}],
    )
    parameters = inspect.signature(tuner.fn).parameters

    before = tuner.generate_cache_key(parameters, {})
    tuner.set_resource_filter_args(True, report_path=str(tmp_path / "resource_filter_report.tsv"), kernel_type="dense_gemm")
    after = tuner.generate_cache_key(parameters, {})

    assert before == after
