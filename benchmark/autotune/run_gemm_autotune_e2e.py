"""Benchmark GEMM autotune end-to-end time across mixed MNK workloads."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.gemm.example_gemm_autotune import run_autotune_with_measurements


DEFAULT_SHAPE_CASES: list[tuple[str, tuple[int, int, int]]] = [
    ("square_large", (4096, 4096, 4096)),
    # ("square_small_k", (4096, 4096, 512)),
    # ("square_large_k", (2048, 2048, 16384)),
    # ("tall_skinny_m", (16384, 1024, 4096)),
    # ("wide_skinny_n", (1024, 16384, 4096)),
    # ("decode_like", (128, 8192, 8192)),
    # ("irregular", (3584, 2816, 1536)),
]


def parse_shape(shape_text: str) -> tuple[int, int, int]:
    parts = shape_text.lower().replace("x", ",").split(",")
    if len(parts) != 3:
        raise ValueError(f"Invalid shape '{shape_text}', expected format like 4096x4096x4096")
    m, n, k = (int(part.strip()) for part in parts)
    return m, n, k


def classify_shape(m: int, n: int, k: int) -> str:
    mn_ratio = max(m, n) / max(1, min(m, n))
    if mn_ratio >= 8:
        return "high_aspect_mn"
    if k <= max(1, min(m, n) // 4):
        return "small_k"
    if k >= 2 * max(m, n):
        return "large_k"
    if any(dim % 128 != 0 for dim in (m, n, k)):
        return "irregular"
    return "balanced"


def run_single_case(
    shape_tag: str,
    shape: tuple[int, int, int],
    cpu_count: int,
    run_idx: int,
    with_roller: bool,
    topk: int,
    config_selector: str,
    selector_pool_k: int | None,
    selector_debug: bool,
    warmup: int,
    rep: int,
    timeout: int,
    profile_backend: str,
    execution_backend: str,
    skip_check: bool,
    cache_input_tensors: bool,
    use_pipeline: bool,
    enable_grouped_compile: bool,
    group_compile_size: int,
    detailed_measurements: bool,
    benchmark_multi_gpu: bool,
    benchmark_devices: list[int],
) -> dict[str, Any]:
    m, n, k = shape

    previous_cpu_count = os.environ.get("TILELANG_AUTO_TUNING_CPU_COUNTS")
    os.environ["TILELANG_AUTO_TUNING_CPU_COUNTS"] = str(cpu_count)

    try:
        _, metrics = run_autotune_with_measurements(
            M=m,
            N=n,
            K=k,
            with_roller=with_roller,
            topk=topk,
            config_selector=config_selector,
            selector_pool_k=selector_pool_k,
            selector_debug=selector_debug,
            warmup=warmup,
            rep=rep,
            timeout=timeout,
            profile_backend=profile_backend,
            execution_backend=execution_backend,
            skip_check=skip_check,
            cache_input_tensors=cache_input_tensors,
            use_pipeline=use_pipeline,
            enable_grouped_compile=enable_grouped_compile,
            group_compile_size=group_compile_size,
            detailed_measurements=detailed_measurements,
            benchmark_multi_gpu=benchmark_multi_gpu,
            benchmark_devices=benchmark_devices,
        )
    finally:
        if previous_cpu_count is None:
            os.environ.pop("TILELANG_AUTO_TUNING_CPU_COUNTS", None)
        else:
            os.environ["TILELANG_AUTO_TUNING_CPU_COUNTS"] = previous_cpu_count

    best_config = metrics.get("best_config")
    shape_text = f"{m}x{n}x{k}"

    row: dict[str, Any] = {
        "shape_tag": shape_tag,
        "shape_class": classify_shape(m, n, k),
        "shape": shape_text,
        "problem_gflop": 2 * m * n * k * 1e-9,
        "cpu_count": cpu_count,
        "run_idx": run_idx,
        "status": metrics.get("status", "failed"),
        "error": metrics.get("error", ""),
        "num_configs": metrics.get("num_configs"),
        "end_to_end_s": metrics.get("end_to_end_s"),
        "compilation_s": metrics.get("compilation_s"),
        "benchmark_s": metrics.get("benchmark_s"),
        "best_latency_ms": metrics.get("best_latency_ms"),
        "best_tflops": metrics.get("best_tflops"),
        "ref_latency_ms": metrics.get("ref_latency_ms"),
        "ref_tflops": metrics.get("ref_tflops"),
        "cpu_count_env": metrics.get("cpu_count_env"),
        "execution_backend": execution_backend,
        "profile_backend": profile_backend,
        "warmup": warmup,
        "rep": rep,
        "timeout": timeout,
        "with_roller": with_roller,
        "topk": topk,
        "selector_name": metrics.get("selector_name", "none"),
        "selector_input_count": metrics.get("selector_input_count"),
        "selector_output_count": metrics.get("selector_output_count"),
        "selector_pool_k": metrics.get("selector_pool_k"),
        "selector_time_ms": metrics.get("selector_time_ms"),
        "selector_fallback_used": metrics.get("selector_fallback_used"),
        "skip_check": skip_check,
        "cache_input_tensors": cache_input_tensors,
        "use_pipeline": use_pipeline,
        "enable_grouped_compile": enable_grouped_compile,
        "group_compile_size": group_compile_size,
        "grouped_compile_active": metrics.get("grouped_compile_active"),
        "num_compile_units_submitted": metrics.get("num_compile_units_submitted"),
        "avg_group_size": metrics.get("avg_group_size"),
        "grouped_compile_reason": metrics.get("grouped_compile_reason", ""),
        "benchmark_multi_gpu_requested": metrics.get("benchmark_multi_gpu_requested", benchmark_multi_gpu),
        "benchmark_multi_gpu_active": metrics.get("benchmark_multi_gpu_active"),
        "benchmark_device_count": metrics.get("benchmark_device_count"),
        "benchmark_devices": metrics.get("benchmark_devices", "[]"),
        "benchmark_shard_policy": metrics.get("benchmark_shard_policy", "static"),
        "best_config": json.dumps(best_config, sort_keys=True) if best_config is not None else "",
    }
    if detailed_measurements:
        row["compile_stage_totals_s"] = json.dumps(metrics.get("compile_stage_totals_s", {}), sort_keys=True)
        row["compile_stage_avg_ms"] = json.dumps(metrics.get("compile_stage_avg_ms", {}), sort_keys=True)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/gemm_autotune_e2e.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        help="Shape MxNxK, can be passed multiple times.",
    )
    parser.add_argument(
        "--cpu-count",
        action="append",
        type=int,
        default=[],
        help="Autotune compiler worker count. Repeat to sweep values.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Repeated runs per (shape, cpu-count) case.",
    )
    parser.add_argument(
        "--execution-backend",
        type=str,
        default="auto",
        choices=["auto", "tvm_ffi", "cython", "nvrtc", "torch"],
    )
    parser.add_argument(
        "--profile-backend",
        type=str,
        default="event",
        choices=["event", "cupti", "cudagraph"],
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Top-k configs when --with-roller is enabled.",
    )
    parser.add_argument(
        "--config-selector",
        type=str,
        default="none",
        choices=["none", "hopper_hybrid_v1"],
        help="Optional preselector applied before compile+benchmark.",
    )
    parser.add_argument(
        "--selector-pool-k",
        type=int,
        default=None,
        help="Optional Roller candidate pool size for selector mode (defaults to Hopper heuristic).",
    )
    selector_debug_group = parser.add_mutually_exclusive_group()
    selector_debug_group.add_argument("--selector-debug", dest="selector_debug", action="store_true")
    selector_debug_group.add_argument("--no-selector-debug", dest="selector_debug", action="store_false")
    parser.set_defaults(selector_debug=False)

    roller_group = parser.add_mutually_exclusive_group()
    roller_group.add_argument("--with-roller", dest="with_roller", action="store_true")
    roller_group.add_argument("--without-roller", dest="with_roller", action="store_false")
    parser.set_defaults(with_roller=False)

    check_group = parser.add_mutually_exclusive_group()
    check_group.add_argument("--skip-check", dest="skip_check", action="store_true")
    check_group.add_argument("--enable-check", dest="skip_check", action="store_false")
    parser.set_defaults(skip_check=True)

    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--disable-cache", dest="disable_cache", action="store_true")
    cache_group.add_argument("--enable-cache", dest="disable_cache", action="store_false")
    parser.set_defaults(disable_cache=True)

    tensor_group = parser.add_mutually_exclusive_group()
    tensor_group.add_argument("--cache-input-tensors", dest="cache_input_tensors", action="store_true")
    tensor_group.add_argument("--no-cache-input-tensors", dest="cache_input_tensors", action="store_false")
    parser.set_defaults(cache_input_tensors=True)

    pipeline_group = parser.add_mutually_exclusive_group()
    pipeline_group.add_argument("--pipeline", dest="use_pipeline", action="store_true")
    pipeline_group.add_argument("--no-pipeline", dest="use_pipeline", action="store_false")
    parser.set_defaults(use_pipeline=False)

    grouped_compile_group = parser.add_mutually_exclusive_group()
    grouped_compile_group.add_argument("--grouped-compile", dest="enable_grouped_compile", action="store_true")
    grouped_compile_group.add_argument("--no-grouped-compile", dest="enable_grouped_compile", action="store_false")
    parser.set_defaults(enable_grouped_compile=False)
    parser.add_argument(
        "--group-compile-size",
        type=int,
        default=2,
        help="Number of configs in one compile unit when grouped compile is enabled.",
    )

    detailed_group = parser.add_mutually_exclusive_group()
    detailed_group.add_argument("--detailed-measurements", dest="detailed_measurements", action="store_true")
    detailed_group.add_argument("--no-detailed-measurements", dest="detailed_measurements", action="store_false")
    parser.set_defaults(detailed_measurements=False)

    benchmark_multi_gpu_group = parser.add_mutually_exclusive_group()
    benchmark_multi_gpu_group.add_argument("--benchmark-multi-gpu", dest="benchmark_multi_gpu", action="store_true")
    benchmark_multi_gpu_group.add_argument("--no-benchmark-multi-gpu", dest="benchmark_multi_gpu", action="store_false")
    parser.set_defaults(benchmark_multi_gpu=False)
    parser.add_argument(
        "--benchmark-devices",
        action="append",
        type=int,
        default=[],
        help="Repeatable CUDA device ordinals for benchmark workers (e.g. --benchmark-devices 0 --benchmark-devices 1).",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs <= 0:
        raise ValueError("--runs must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.rep <= 0:
        raise ValueError("--rep must be > 0")
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")
    if args.group_compile_size <= 0:
        raise ValueError("--group-compile-size must be > 0")
    if args.selector_pool_k is not None and args.selector_pool_k <= 0:
        raise ValueError("--selector-pool-k must be > 0 when provided")
    if any(cpu <= 0 for cpu in args.cpu_count):
        raise ValueError("--cpu-count values must be > 0")
    if any(device < 0 for device in args.benchmark_devices):
        raise ValueError("--benchmark-devices values must be >= 0")

    if args.shape:
        shape_cases = [(f"custom_{idx + 1}", parse_shape(shape_text)) for idx, shape_text in enumerate(args.shape)]
    else:
        shape_cases = DEFAULT_SHAPE_CASES
    cpu_counts = args.cpu_count if args.cpu_count else [16]

    if args.disable_cache:
        os.environ["TILELANG_DISABLE_CACHE"] = "1"
        os.environ["TILELANG_AUTO_TUNING_DISABLE_CACHE"] = "1"
    else:
        os.environ["TILELANG_DISABLE_CACHE"] = "0"
        os.environ["TILELANG_AUTO_TUNING_DISABLE_CACHE"] = "0"

    args.csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "shape_tag",
        "shape_class",
        "shape",
        "problem_gflop",
        "cpu_count",
        "run_idx",
        "status",
        "error",
        "num_configs",
        "end_to_end_s",
        "compilation_s",
        "benchmark_s",
        "best_latency_ms",
        "best_tflops",
        "ref_latency_ms",
        "ref_tflops",
        "cpu_count_env",
        "execution_backend",
        "profile_backend",
        "warmup",
        "rep",
        "timeout",
        "with_roller",
        "topk",
        "selector_name",
        "selector_input_count",
        "selector_output_count",
        "selector_pool_k",
        "selector_time_ms",
        "selector_fallback_used",
        "skip_check",
        "cache_input_tensors",
        "use_pipeline",
        "enable_grouped_compile",
        "group_compile_size",
        "grouped_compile_active",
        "num_compile_units_submitted",
        "avg_group_size",
        "grouped_compile_reason",
        "benchmark_multi_gpu_requested",
        "benchmark_multi_gpu_active",
        "benchmark_device_count",
        "benchmark_devices",
        "benchmark_shard_policy",
        "best_config",
    ]
    if args.detailed_measurements:
        fieldnames.insert(fieldnames.index("best_latency_ms"), "compile_stage_totals_s")
        fieldnames.insert(fieldnames.index("best_latency_ms"), "compile_stage_avg_ms")

    total_cases = len(shape_cases) * len(cpu_counts) * args.runs
    print(f"Writing CSV to: {args.csv}")
    print(f"Total cases: {total_cases}")

    completed = 0
    with args.csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for shape_tag, shape in shape_cases:
            for cpu_count in cpu_counts:
                for run_idx in range(1, args.runs + 1):
                    completed += 1
                    m, n, k = shape
                    print(f"[{completed}/{total_cases}] tag={shape_tag} shape={m}x{n}x{k} cpu_count={cpu_count} run={run_idx}")
                    row = run_single_case(
                        shape_tag=shape_tag,
                        shape=shape,
                        cpu_count=cpu_count,
                        run_idx=run_idx,
                        with_roller=args.with_roller,
                        topk=args.topk,
                        config_selector=args.config_selector,
                        selector_pool_k=args.selector_pool_k,
                        selector_debug=args.selector_debug,
                        warmup=args.warmup,
                        rep=args.rep,
                        timeout=args.timeout,
                        profile_backend=args.profile_backend,
                        execution_backend=args.execution_backend,
                        skip_check=args.skip_check,
                        cache_input_tensors=args.cache_input_tensors,
                        use_pipeline=args.use_pipeline,
                        enable_grouped_compile=args.enable_grouped_compile,
                        group_compile_size=args.group_compile_size,
                        detailed_measurements=args.detailed_measurements,
                        benchmark_multi_gpu=args.benchmark_multi_gpu,
                        benchmark_devices=args.benchmark_devices,
                    )
                    writer.writerow(row)
                    fp.flush()
                    print(
                        f"[{completed}/{total_cases}] status={row['status']} "
                        f"end_to_end_s={row['end_to_end_s']:.3f} "
                        f"best_latency_ms={row['best_latency_ms']}"
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
