"""Reproducible cold-cache comparison of new Carver on a simple GEMM.

Bind an idle device with CUDA_VISIBLE_DEVICES=N.
Each invocation tests one mode in a fresh process and emits all config outcomes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import itertools
import hashlib
import os
from pathlib import Path
import subprocess
import time

# Apply before importing TileLang, including its CUDA binary cache.
os.environ["TILELANG_DISABLE_CACHE"] = "1"
os.environ["TILELANG_AUTO_TUNING_DISABLE_CACHE"] = "1"
os.environ["TILELANG_AUTO_TUNING_CPU_COUNTS"] = "4"

import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import AutoTuner
from tilelang.new_carver.config import ANALYSIS_VERSION
from tilelang.new_carver.device_profile import current_target, _identity, load_device_profile


TILES = [(64, 64), (64, 128), (128, 128), (128, 256), (256, 256)]
CONFIGS = [{"bm": m, "bn": n, "threads": threads} for m, n in TILES for threads in (128, 256)]


def make_gemm(size):
    def gemm(bm=64, bn=64, threads=128):
        @T.prim_func
        def main(A: T.Tensor((size, size), "float16"), B: T.Tensor((size, size), "float16"), C: T.Tensor((size, size), "float32")):
            with T.Kernel(T.ceildiv(size, bm), T.ceildiv(size, bn), threads=threads) as (bx, by):
                a = T.alloc_shared((bm, 32), "float16")
                b = T.alloc_shared((32, bn), "float16")
                c = T.alloc_fragment((bm, bn), "float32")
                T.clear(c)
                for k in T.Pipelined(T.ceildiv(size, 32), num_stages=2):
                    T.copy(A[bx * bm, k * 32], a)
                    T.copy(B[k * 32, by * bn], b)
                    T.gemm(a, b, c)
                T.copy(c, C[bx * bm, by * bn])

        return main

    return gemm


def full_configs():
    keys = ("block_M", "block_N", "block_K", "num_stages", "thread_num", "enable_rasteration")
    values = itertools.product([64, 128, 256], [64, 128, 256], [32, 64], [0, 1, 2, 3], [128, 256], [True, False])
    configs = [dict(zip(keys, item)) for item in values]
    assert len(configs) == 288
    return configs


def make_full_gemm(size):
    def gemm(block_M, block_N, block_K, num_stages, thread_num, enable_rasteration):
        @T.prim_func
        def main(A: T.Tensor((size, size), "float16"), B: T.Tensor((size, size), "float16"), C: T.Tensor((size, size), "float32")):
            with T.Kernel(T.ceildiv(size, block_M), T.ceildiv(size, block_N), threads=thread_num) as (bx, by):
                a = T.alloc_shared((block_M, block_K), "float16")
                b = T.alloc_shared((block_K, block_N), "float16")
                c = T.alloc_fragment((block_M, block_N), "float32")
                T.use_swizzle(panel_size=10, enable=enable_rasteration)
                T.clear(c)
                for k in T.Pipelined(T.ceildiv(size, block_K), num_stages=num_stages):
                    T.copy(A[bx * block_M, k * block_K], a)
                    T.copy(B[k * block_K, by * block_N], b)
                    T.gemm(a, b, c)
                T.copy(c, C[bx * block_M, by * block_N])

        return main

    return gemm


class ObservedTuner(AutoTuner):
    """Record outcomes without changing candidate compilation or benchmarking."""

    def _prepare_compile_execution(self, **kwargs):
        result = super()._prepare_compile_execution(**kwargs)
        for future in result[1]:

            def record(done):
                try:
                    for idx, _config, kernel, error in done.result():
                        row = self.outcomes[idx]
                        row["compile_status"] = "compiled" if error is None else "failed_or_rejected"
                        row["compile_error"] = str(error) if error else None
                        if kernel is not None:
                            row["resources"] = {name: asdict(item) for name, item in getattr(kernel, "_resource_usage", {}).items()}
                            source = kernel.get_kernel_source()
                            row["instruction"] = "wgmma" if "tl::wgmma" in source else "mma"
                            (self.run_dir / f"config_{idx}.cu").write_text(source)
                except Exception as error:
                    self.record_errors.append(str(error))

            future.add_done_callback(record)
        return result

    def _write_benchmark_result(self, idx, status, latency, error):
        super()._write_benchmark_result(idx, status, latency, error)
        self.outcomes[idx].update(benchmark_status=status, latency_ms=latency, benchmark_error=error)


def snapshot(gpu=None):
    if gpu is None:
        gpu = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
        if not gpu.startswith("GPU-"):
            gpu = "GPU-" + gpu
    return subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=index,uuid,name,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["disabled", "report_only", "reject"], required=True)
    parser.add_argument("--gpu", type=int, help="Physical GPU label (visibility is set by the wrapper)")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--space", choices=["smoke", "full288", "calibration128"], default="smoke")
    parser.add_argument("--ranking-metric", choices=["traffic_waves", "pipeline_time"], default="traffic_waves")
    parser.add_argument("--performance-profile", type=Path, help="Explicit effective-rate profile JSON")
    parser.add_argument("--device-profile", type=Path, help="Reusable device profile; use streaming rates")
    parser.add_argument("--compile-workers", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["TILELANG_AUTO_TUNING_CPU_COUNTS"] = str(args.compile_workers)
    configs = full_configs() if args.space != "smoke" else CONFIGS
    if args.space == "calibration128":
        configs = [c for c in configs if c["block_M"] <= 128 and c["block_N"] <= 128]
        assert len(configs) == 128
    factory = make_full_gemm if args.space != "smoke" else make_gemm
    profile = json.loads(args.performance_profile.read_text()) if args.performance_profile else None
    if args.device_profile:
        if args.performance_profile:
            parser.error("choose --device-profile or --performance-profile")
        profile = load_device_profile(args.device_profile, input_dtype="float16", expected_identity=_identity(), memory_regime="streaming")
    run_dir = args.output / f"gpu{args.gpu}_{args.mode}_{args.repeat}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "configs.json").write_text(json.dumps(configs, indent=2) + "\n")
    os.environ["TILELANG_AUTOTUNE_TIMING_LOG"] = str(run_dir / "timings.tsv")
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    a = torch.rand((args.size, args.size), device="cuda", dtype=torch.float16) - 0.5
    b = torch.rand_like(a) - 0.5
    reference = a.float() @ b.float()
    torch.cuda.synchronize()
    inputs = [a, b]
    tuner = (
        ObservedTuner(factory(args.size), configs)
        .set_compile_args(
            target=current_target(),
            execution_backend="tvm_ffi",
            out_idx=[2],
            pass_configs={"tl.enable_cuda_resource_capture": True} if args.mode == "disabled" else None,
        )
        .set_profile_args(
            ref_prog=lambda x, y: x.float() @ y.float(),
            supply_prog=lambda params: inputs,
            backend="cudagraph",
            rtol=0.01,
            atol=0.01,
            max_mismatched_ratio=0.0,
            cache_input_tensors=True,
        )
        .set_benchmark_report_path(str(run_dir / "benchmarks.tsv"))
    )
    tuner.run_dir = run_dir
    tuner.record_errors = []
    tuner.outcomes = [{"index": i, "config": c, "compile_status": "pending", "benchmark_status": "not_run"} for i, c in enumerate(configs)]
    if args.mode != "disabled":
        tuner.set_carver_args(
            True,
            mode=args.mode,
            report_path=str(run_dir / "carver.json"),
            ranking_metric=args.ranking_metric,
            performance_model=profile,
        )
    before = snapshot(args.gpu)
    start = time.perf_counter()
    result = None
    error = None
    try:
        result = tuner.run(warmup=args.warmup, rep=args.rep, early_stop=False, use_pipeline=False)
    except Exception as exc:
        error = str(exc)
    elapsed = time.perf_counter() - start
    after = snapshot(args.gpu)
    winner_latency = None
    winner_samples = []
    winner_accuracy = None
    if result is not None:
        actual = result.kernel(a, b)
        torch.testing.assert_close(actual, reference, rtol=0.01, atol=0.01)
        difference = (actual - reference).abs()
        winner_accuracy = {
            "max_abs_error": difference.max().item(),
            "rmse": difference.square().mean().sqrt().item(),
            "relative_l2_error": (difference.norm() / reference.norm()).item(),
            "tolerance_violations": (difference > 0.01 + 0.01 * reference.abs()).sum().item(),
            "elements": actual.numel(),
        }
        # Separate repeated winner measurements from the tuning selection sample.
        profiler = result.kernel.get_profiler()
        winner_samples = [profiler.do_bench(input_tensors=inputs, n_warmup=20, n_repeat=200, backend="cudagraph") for _ in range(5)]
        winner_latency = sorted(winner_samples)[len(winner_samples) // 2]
    summary = {
        "mode": args.mode,
        "gpu": args.gpu,
        "repeat": args.repeat,
        "size": args.size,
        "cache_disabled": True,
        "compile_workers": args.compile_workers,
        "config_space": args.space,
        "config_count": len(configs),
        "warmup": args.warmup,
        "rep": args.rep,
        "grouped_compile": False,
        "torch_version": torch.__version__,
        "tilelang_version": tilelang.__version__,
        "analysis_version": ANALYSIS_VERSION,
        "ranking_metric": args.ranking_metric,
        "performance_model": profile,
        "analysis_source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(Path("tilelang/new_carver").glob("*.py"))
        },
        "gpu_before": before,
        "gpu_after": after,
        "wall_s": elapsed,
        "error": error,
        "winner_config": result.config if result else None,
        "tuning_latency_ms": result.latency if result else None,
        "winner_latency_ms": winner_latency,
        "winner_samples_ms": winner_samples,
        "winner_accuracy": winner_accuracy,
        "configs": tuner.outcomes,
        "record_errors": tuner.record_errors,
        "carver": tuner.carver_report,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in ("mode", "gpu", "repeat", "wall_s", "error", "winner_config", "winner_latency_ms")}))
    if error or tuner.record_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
