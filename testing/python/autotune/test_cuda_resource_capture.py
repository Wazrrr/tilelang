from __future__ import annotations

import importlib

import tilelang
from tilelang import tvm
from tilelang.autotuner import AutoTuner
from tilelang.autotuner.filters import (
    extract_launch_resource_info,
)
from tilelang.cache.cuda_binary_cache import CUDABinaryCache
from tilelang.cache.kernel_cache import KernelCache
from tilelang.contrib.cuda_resource_info import CUDA_RESOURCE_CAPTURE_CONFIG_KEY
from tilelang.contrib.cuda_resource_info import pop_recorded as cuda_pop_recorded
from tilelang.contrib.cuda_resource_info import reset_recorder as cuda_reset_recorder
from tilelang.env import env
from tvm.target import Target


def _set_cache_dirs(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    tmp_dir = tmp_path / "tmp"
    cache_dir.mkdir()
    tmp_dir.mkdir()
    # TILELANG_TMP_DIR's default is derived from TILELANG_CACHE_DIR. Patch it
    # first so pytest records the real original value for teardown.
    monkeypatch.setattr(env, "TILELANG_TMP_DIR", str(tmp_dir))
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(env, "TILELANG_DISABLE_CACHE", "0")
    tilelang.enable_cache()
    AutoTuner._memory_cache.clear()
    KernelCache._get_cache_namespace.cache_clear()
    CUDABinaryCache._get_tilelang_lib_stamp.cache_clear()


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
