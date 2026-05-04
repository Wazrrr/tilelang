"""Debug runner to inspect TVM FFI prepare vs kernel timing during GEMM autotune."""

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


def parse_shape(shape_text: str) -> tuple[int, int, int]:
    parts = shape_text.lower().replace("x", ",").split(",")
    if len(parts) != 3:
        raise ValueError(f"Invalid shape '{shape_text}', expected format like 4096x4096x4096")
    m, n, k = (int(part.strip()) for part in parts)
    return m, n, k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        action="append",
        default=["4096x4096x4096"],
        help="Shape MxNxK. Repeat for multiple shapes.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/tvm_ffi_timing_debug.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--execution-backend",
        type=str,
        default="auto",
        choices=["auto", "tvm_ffi", "cython", "nvrtc", "torch"],
        help="Use auto/tvm_ffi for this timing debug.",
    )
    parser.add_argument(
        "--profile-backend",
        type=str,
        default="event",
        choices=["event", "cupti", "cudagraph"],
    )
    parser.add_argument("--cpu-count", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--timing-print-limit",
        type=int,
        default=1,
        help="Passed to TL_TVM_FFI_TIMING_PRINT_LIMIT.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help="Optional override for CUDA_VISIBLE_DEVICES.",
    )

    roller_group = parser.add_mutually_exclusive_group()
    roller_group.add_argument("--with-roller", dest="with_roller", action="store_true")
    roller_group.add_argument("--without-roller", dest="with_roller", action="store_false")
    parser.set_defaults(with_roller=True)

    pipeline_group = parser.add_mutually_exclusive_group()
    pipeline_group.add_argument("--pipeline", dest="use_pipeline", action="store_true")
    pipeline_group.add_argument("--no-pipeline", dest="use_pipeline", action="store_false")
    parser.set_defaults(use_pipeline=False)

    grouped_compile_group = parser.add_mutually_exclusive_group()
    grouped_compile_group.add_argument("--grouped-compile", dest="enable_grouped_compile", action="store_true")
    grouped_compile_group.add_argument("--no-grouped-compile", dest="enable_grouped_compile", action="store_false")
    parser.set_defaults(enable_grouped_compile=False)
    parser.add_argument("--group-compile-size", type=int, default=2)

    check_group = parser.add_mutually_exclusive_group()
    check_group.add_argument("--skip-check", dest="skip_check", action="store_true")
    check_group.add_argument("--enable-check", dest="skip_check", action="store_false")
    parser.set_defaults(skip_check=True)

    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--cache-input-tensors", dest="cache_input_tensors", action="store_true")
    cache_group.add_argument("--no-cache-input-tensors", dest="cache_input_tensors", action="store_false")
    parser.set_defaults(cache_input_tensors=True)

    disable_cache_group = parser.add_mutually_exclusive_group()
    disable_cache_group.add_argument("--disable-cache", dest="disable_cache", action="store_true")
    disable_cache_group.add_argument("--enable-cache", dest="disable_cache", action="store_false")
    parser.set_defaults(disable_cache=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from examples.gemm.example_gemm_autotune import run_autotune_with_measurements

    if args.cpu_count <= 0:
        raise ValueError("--cpu-count must be > 0")
    if args.runs <= 0:
        raise ValueError("--runs must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.rep <= 0:
        raise ValueError("--rep must be > 0")
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")
    if args.topk <= 0:
        raise ValueError("--topk must be > 0")
    if args.group_compile_size <= 0:
        raise ValueError("--group-compile-size must be > 0")
    if args.timing_print_limit < 0:
        raise ValueError("--timing-print-limit must be >= 0")

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["TILELANG_AUTO_TUNING_CPU_COUNTS"] = str(args.cpu_count)
    os.environ["TL_TVM_FFI_TIMING_PRINT_LIMIT"] = str(args.timing_print_limit)
    if args.disable_cache:
        os.environ["TILELANG_DISABLE_CACHE"] = "1"
        os.environ["TILELANG_AUTO_TUNING_DISABLE_CACHE"] = "1"
    else:
        os.environ["TILELANG_DISABLE_CACHE"] = "0"
        os.environ["TILELANG_AUTO_TUNING_DISABLE_CACHE"] = "0"

    shapes = [parse_shape(s) for s in args.shape]
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "shape",
        "run_idx",
        "status",
        "error",
        "execution_backend",
        "profile_backend",
        "num_configs",
        "end_to_end_s",
        "compilation_s",
        "benchmark_s",
        "compile_stage_totals_s",
        "compile_stage_avg_ms",
        "best_latency_ms",
        "ref_latency_ms",
        "best_tflops",
        "ref_tflops",
        "timing_print_limit",
        "cpu_count",
        "with_roller",
        "topk",
        "use_pipeline",
        "enable_grouped_compile",
        "group_compile_size",
        "best_config",
    ]

    print(f"Writing CSV to: {args.csv}")
    print(
        "Environment: "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
        f"TILELANG_AUTO_TUNING_CPU_COUNTS={os.environ.get('TILELANG_AUTO_TUNING_CPU_COUNTS')} "
        f"TL_TVM_FFI_TIMING_PRINT_LIMIT={os.environ.get('TL_TVM_FFI_TIMING_PRINT_LIMIT')}"
    )
    print("Expect per-call debug lines from tvm_ffi adapter: [TVMFFI][timing] ...")

    with args.csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        total_cases = len(shapes) * args.runs
        case_idx = 0
        for m, n, k in shapes:
            for run_idx in range(1, args.runs + 1):
                case_idx += 1
                shape_text = f"{m}x{n}x{k}"
                print(f"[{case_idx}/{total_cases}] shape={shape_text} run={run_idx}")
                _result, metrics = run_autotune_with_measurements(
                    M=m,
                    N=n,
                    K=k,
                    with_roller=args.with_roller,
                    profile_backend=args.profile_backend,
                    execution_backend=args.execution_backend,
                    warmup=args.warmup,
                    rep=args.rep,
                    timeout=args.timeout,
                    skip_check=args.skip_check,
                    cache_input_tensors=args.cache_input_tensors,
                    topk=args.topk,
                    use_pipeline=args.use_pipeline,
                    enable_grouped_compile=args.enable_grouped_compile,
                    group_compile_size=args.group_compile_size,
                )
                row: dict[str, Any] = {
                    "shape": shape_text,
                    "run_idx": run_idx,
                    "status": metrics.get("status"),
                    "error": metrics.get("error", ""),
                    "execution_backend": args.execution_backend,
                    "profile_backend": args.profile_backend,
                    "num_configs": metrics.get("num_configs"),
                    "end_to_end_s": metrics.get("end_to_end_s"),
                    "compilation_s": metrics.get("compilation_s"),
                    "benchmark_s": metrics.get("benchmark_s"),
                    "compile_stage_totals_s": json.dumps(metrics.get("compile_stage_totals_s", {}), sort_keys=True),
                    "compile_stage_avg_ms": json.dumps(metrics.get("compile_stage_avg_ms", {}), sort_keys=True),
                    "best_latency_ms": metrics.get("best_latency_ms"),
                    "ref_latency_ms": metrics.get("ref_latency_ms"),
                    "best_tflops": metrics.get("best_tflops"),
                    "ref_tflops": metrics.get("ref_tflops"),
                    "timing_print_limit": args.timing_print_limit,
                    "cpu_count": args.cpu_count,
                    "with_roller": args.with_roller,
                    "topk": args.topk,
                    "use_pipeline": args.use_pipeline,
                    "enable_grouped_compile": args.enable_grouped_compile,
                    "group_compile_size": args.group_compile_size,
                    "best_config": json.dumps(metrics.get("best_config"), sort_keys=True)
                    if metrics.get("best_config") is not None
                    else "",
                }
                writer.writerow(row)
                fp.flush()
                print(
                    "  result: "
                    f"status={row['status']} end_to_end_s={row['end_to_end_s']} "
                    f"compilation_s={row['compilation_s']} benchmark_s={row['benchmark_s']} "
                    f"best_latency_ms={row['best_latency_ms']}"
                )
                print(f"  compile_stage_avg_ms={row['compile_stage_avg_ms']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
