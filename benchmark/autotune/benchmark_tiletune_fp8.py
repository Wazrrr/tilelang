"""Exhaustive FP8 validation of a previously frozen, candidate-independent profile."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

# The observation harness also disables JIT/autotuner caches before importing TL.
from benchmark.autotune.benchmark_tiletune_gemm import ObservedTuner, snapshot
from examples.gemm_fp8.example_gemm_fp8_tiletune import get_configs, make_gemm, TARGET
from examples.gemm_fp8.example_tilelang_gemm_fp8 import calc_diff
import torch
import tilelang.language as T
from tilelang.tiletune import load_device_profile
from tilelang.tiletune.config import ANALYSIS_VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--device-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", default="float8_e4m3fn", choices=["float8_e4m3fn", "float8_e5m2"])
    parser.add_argument("--mode", default="report_only", choices=["disabled", "report_only", "reject"])
    parser.add_argument("--indices", help="Optional comma-separated original indices for a smoke check")
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=1)
    args = parser.parse_args()
    os.environ["TILELANG_AUTO_TUNING_CPU_COUNTS"] = str(args.workers)
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["TILELANG_AUTOTUNE_TIMING_LOG"] = str(args.output / "timings.tsv")
    configs = get_configs()
    indices = list(map(int, args.indices.split(","))) if args.indices else list(range(len(configs)))
    configs = [configs[i] for i in indices]
    profile = load_device_profile(args.device_profile, input_dtype=args.dtype)
    # Persist exactly what the ranker knew before any candidate was benchmarked.
    frozen = dict(
        profile=profile,
        source_sha256={
            p.relative_to("tilelang/tiletune").as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("tilelang/tiletune").rglob("*.py")
        },
    )
    (args.output / "frozen.json").write_text(json.dumps(frozen, indent=2) + "\n")
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = T.dtype(args.dtype).as_torch()
    inputs = [(torch.rand((args.size, args.size), device="cuda", dtype=torch.float16) - 0.5).to(dtype) for _ in range(2)]
    reference = (inputs[0].float() @ inputs[1].float().T).to(dtype)
    accuracies = []

    def check(actuals, refs):
        error = calc_diff(actuals[0], refs[0]).item()
        accuracies.append(error)
        assert error < 1e-3, f"FP8 example calc_diff={error} >= 1e-3"

    tuner = (
        ObservedTuner(make_gemm(args.size, args.size, args.size, args.dtype), configs)
        .set_compile_args(target=TARGET, execution_backend="tvm_ffi")
        .set_profile_args(
            ref_prog=lambda a, b: reference,
            supply_prog=lambda params: inputs,
            manual_check_prog=check,
            backend="cudagraph",
            cache_input_tensors=True,
        )
        .set_benchmark_report_path(str(args.output / "benchmarks.tsv"))
    )
    if args.mode != "disabled":
        tuner.set_tiletune_args(
            True, mode=args.mode, report_path=str(args.output / "tiletune.json"), ranking_metric="pipeline_time", performance_model=profile
        )
    tuner.run_dir = args.output
    tuner.record_errors = []
    tuner.outcomes = [
        dict(index=i, original_index=indices[i], config=c, compile_status="pending", benchmark_status="not_run")
        for i, c in enumerate(configs)
    ]
    before, start = snapshot(args.gpu), time.perf_counter()
    result, error = None, None
    try:
        result = tuner.run(
            warmup=10,
            rep=100,
            early_stop=False,
            use_pipeline=False,
            enable_grouped_compile=args.group_size > 1,
            group_compile_size=args.group_size,
        )
    except Exception as exc:
        error = str(exc)
    summary = dict(
        dtype=args.dtype,
        size=args.size,
        mode=args.mode,
        gpu=args.gpu,
        analysis_version=ANALYSIS_VERSION,
        config_count=len(configs),
        group_size=args.group_size,
        wall_s=time.perf_counter() - start,
        gpu_before=before,
        gpu_after=snapshot(args.gpu),
        error=error,
        profile=profile,
        configs=tuner.outcomes,
        tiletune=tuner.tiletune_report,
        record_errors=tuner.record_errors,
        correctness=dict(
            metric="FP8 example calc_diff",
            threshold=1e-3,
            checks=len(accuracies),
            max_error=max(accuracies) if accuracies else None,
            errors_in_benchmark_order=accuracies,
        ),
        winner_config=result.config if result else None,
        winner_latency_ms=result.latency if result else None,
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in ("config_count", "wall_s", "error", "correctness", "winner_config", "winner_latency_ms")}))
    if error or tuner.record_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
