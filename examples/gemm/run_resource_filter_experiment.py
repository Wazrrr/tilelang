from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from collections import Counter
from pathlib import Path

import tilelang
import tilelang.language as T
from tilelang.autotuner import AutoTuner


def make_matmul_kernel(M: int, N: int, K: int, dtype):
    def kernel(
        block_M=None,
        block_N=None,
        block_K=None,
        num_stages=1,
        thread_num=None,
        enable_rasteration=False,
    ):
        @T.prim_func
        def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                C_local = T.alloc_fragment((block_M, block_N), T.float32)
                C_shared = T.alloc_shared((block_M, block_N), dtype)
                T.use_swizzle(panel_size=10, enable=enable_rasteration)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])

        return main

    return kernel


def make_configs(full_configs: bool, include_invalid_extra: bool) -> list[dict]:
    if full_configs:
        configs = [
            {
                "block_M": block_m,
                "block_N": block_n,
                "block_K": block_k,
                "num_stages": num_stages,
                "thread_num": thread_num,
                "enable_rasteration": enable_rasteration,
            }
            for block_m, block_n, block_k, num_stages, thread_num, enable_rasteration in itertools.product(
                [64, 128, 256],
                [64, 128, 256],
                [32, 64],
                [0, 1, 2, 3],
                [128, 256],
                [True, False],
            )
        ]
        assert len(configs) == 288
    else:
        configs = [
            {
                "block_M": 64,
                "block_N": 64,
                "block_K": 32,
                "thread_num": 128,
            },
            {
                "block_M": 64,
                "block_N": 64,
                "block_K": 64,
                "thread_num": 128,
            },
        ]

    if include_invalid_extra:
        configs.append(
            {
                "block_M": 64,
                "block_N": 64,
                "block_K": 65536,
                "thread_num": 128,
            }
        )
    return configs


def summarize_report(path: Path) -> dict[str, object]:
    verdict_counts: Counter[str] = Counter()
    stage_verdict_counts: Counter[tuple[str, str]] = Counter()
    reason_counts: Counter[tuple[str, str, str]] = Counter()
    violation_counts: Counter[str] = Counter()
    advisory_counts: Counter[str] = Counter()
    if not path.exists():
        return {
            "verdicts": verdict_counts,
            "stage_verdicts": stage_verdict_counts,
            "reasons": reason_counts,
            "violations": violation_counts,
            "advisories": advisory_counts,
            "rows": 0,
        }

    with path.open(newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        rows = 0
        for row in reader:
            rows += 1
            stage = row.get("stage", "")
            verdict = row.get("verdict", "")
            reason = row.get("reason", "")
            verdict_counts[verdict] += 1
            stage_verdict_counts[(stage, verdict)] += 1
            reason_counts[(stage, verdict, reason)] += 1
            try:
                details = json.loads(row.get("details", "") or "{}")
            except json.JSONDecodeError:
                details = {}
            for violation in details.get("violations", []) or []:
                violation_counts[str(violation.get("reason", ""))] += 1
            for advisory in details.get("advisories", []) or []:
                advisory_counts[str(advisory.get("reason", ""))] += 1
    return {
        "verdicts": verdict_counts,
        "stage_verdicts": stage_verdict_counts,
        "reasons": reason_counts,
        "violations": violation_counts,
        "advisories": advisory_counts,
        "rows": rows,
    }


def summarize_benchmark_report(path: Path | None) -> dict[str, object]:
    status_counts: Counter[str] = Counter()
    best_latency = None
    best_config = None
    best_index = None
    rows = 0
    if path is None or not path.exists():
        return {
            "statuses": status_counts,
            "rows": rows,
            "best_latency": best_latency,
            "best_config": best_config,
            "best_index": best_index,
        }

    with path.open(newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            rows += 1
            status = row.get("status", "")
            status_counts[status] += 1
            if status != "ok" or not row.get("latency_ms"):
                continue
            latency = float(row["latency_ms"])
            if best_latency is None or latency < best_latency:
                best_latency = latency
                best_config = json.loads(row["config"])
                best_index = int(row["index"])
    return {
        "statuses": status_counts,
        "rows": rows,
        "best_latency": best_latency,
        "best_config": best_config,
        "best_index": best_index,
    }


def dtype_from_name(name: str):
    if name == "float16":
        return T.float16
    if name == "bfloat16":
        return T.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def format_counter(counter: Counter) -> str:
    if not counter:
        return "{}"
    return ", ".join(f"{key}={value}" for key, value in sorted(counter.items(), key=lambda item: str(item[0])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CUDA GEMM autotune experiment with exact resource filtering.")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--report", type=Path, default=Path("resource_filter_report.tsv"))
    parser.add_argument("--quality-filter", action="store_true", help="Enable exact post-compile CUDA quality filtering.")
    parser.add_argument("--quality-report", type=Path, default=Path("quality_filter_report.tsv"))
    parser.add_argument("--quality-action", choices=["reject", "report"], default="reject")
    parser.add_argument("--quality-max-spills", type=int, default=0)
    parser.add_argument("--quality-max-local-bytes", type=int, default=0)
    parser.add_argument("--quality-max-c-local", type=int, default=256)
    parser.add_argument("--quality-max-output-per-thread", type=int, default=256)
    parser.add_argument("--quality-max-wgmma-n", type=int, default=None)
    parser.add_argument("--quality-max-k-loop", type=int, default=None)
    parser.add_argument("--quality-advisory-max-wgmma-n", type=int, default=128)
    parser.add_argument("--quality-advisory-max-k-loop", type=int, default=64)
    parser.add_argument("--perf-report", type=Path, default=None, help="Optional TSV path for per-config benchmark results.")
    parser.add_argument("--full-configs", action="store_true", help="Use the full 288-config GEMM search space.")
    parser.add_argument("--grouped", action="store_true", help="Use grouped CUDA+tvm_ffi compilation.")
    parser.add_argument("--group-compile-size", type=int, default=16, help="Configs per grouped compile unit.")
    parser.add_argument("--include-invalid-extra", action="store_true", help="Append one intentionally invalid config.")
    parser.add_argument("--disable-cache", action="store_true", help="Disable TileLang cache for this run.")
    args = parser.parse_args()
    start = time.perf_counter()

    if args.disable_cache:
        tilelang.disable_cache()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    kernel = make_matmul_kernel(args.m, args.n, args.k, dtype_from_name(args.dtype))
    configs = make_configs(full_configs=args.full_configs, include_invalid_extra=args.include_invalid_extra)
    print(f"config_count={len(configs)}")

    tuner = (
        AutoTuner.from_kernel(kernel=kernel, configs=configs)
        .set_compile_args(out_idx=[-1], target="cuda", execution_backend="tvm_ffi")
        .set_profile_args(supply_type=tilelang.TensorSupplyType.Integer, skip_check=True)
        .set_resource_filter_args(True, report_path=str(args.report))
        .set_benchmark_report_path(args.perf_report)
    )
    if args.quality_filter:
        tuner.set_quality_filter_args(
            True,
            action=args.quality_action,
            report_path=str(args.quality_report),
            max_spills=args.quality_max_spills,
            max_local_size_bytes=args.quality_max_local_bytes,
            max_c_local_floats=args.quality_max_c_local,
            max_output_elements_per_thread=args.quality_max_output_per_thread,
            max_wgmma_n=args.quality_max_wgmma_n,
            max_k_loop_iterations=args.quality_max_k_loop,
            advisory_max_wgmma_n=args.quality_advisory_max_wgmma_n,
            advisory_max_k_loop_iterations=args.quality_advisory_max_k_loop,
        )
    result = tuner.run(
        warmup=args.warmup,
        rep=args.rep,
        enable_grouped_compile=args.grouped,
        group_compile_size=args.group_compile_size,
    )

    counts = summarize_report(args.report)
    print(f"best_config={result.config}")
    print(f"best_latency_ms={result.latency}")
    print(f"filter_report={args.report.resolve()}")
    print(f"elapsed_s={time.perf_counter() - start:.3f}")
    print(f"filter_report_rows={counts['rows']}")
    print(f"filter_verdicts={format_counter(counts['verdicts'])}")
    print(f"filter_stage_verdicts={format_counter(counts['stage_verdicts'])}")
    print(f"filter_reasons={format_counter(counts['reasons'])}")
    if args.quality_filter:
        quality_counts = summarize_report(args.quality_report)
        print(f"quality_filter_report={args.quality_report.resolve()}")
        print(f"quality_filter_report_rows={quality_counts['rows']}")
        print(f"quality_filter_verdicts={format_counter(quality_counts['verdicts'])}")
        print(f"quality_filter_stage_verdicts={format_counter(quality_counts['stage_verdicts'])}")
        print(f"quality_filter_reasons={format_counter(quality_counts['reasons'])}")
        print(f"quality_filter_violations={format_counter(quality_counts['violations'])}")
        print(f"quality_filter_advisories={format_counter(quality_counts['advisories'])}")
    if args.perf_report is not None:
        perf_counts = summarize_benchmark_report(args.perf_report)
        print(f"benchmark_report={args.perf_report.resolve()}")
        print(f"benchmark_report_rows={perf_counts['rows']}")
        print(f"benchmark_statuses={format_counter(perf_counts['statuses'])}")
        print(f"benchmark_report_best_index={perf_counts['best_index']}")
        print(f"benchmark_report_best_latency_ms={perf_counts['best_latency']}")
        print(f"benchmark_report_best_config={perf_counts['best_config']}")


if __name__ == "__main__":
    main()
