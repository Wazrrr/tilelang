from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import tilelang
import tilelang.language as T
from tilelang.autotuner import AutoTuner
from tilelang.autotuner.tuner import _PASS_CONFIGS_KEY, _normalize_value


def make_dense_gemm_kernel(M: int, N: int, K: int, dtype):
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


def dense_gemm_configs() -> list[dict[str, Any]]:
    return [
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


def _add_example_dir(name: str) -> None:
    path = Path(__file__).resolve().parents[1] / name
    sys.path.insert(0, str(path))


def _format_counter(counter: Counter) -> str:
    if not counter:
        return "{}"
    return ", ".join(f"{key}={value}" for key, value in sorted(counter.items(), key=lambda item: str(item[0])))


def _read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def _json_cell(row: dict[str, str], key: str) -> Any:
    try:
        return json.loads(row.get(key, "") or "{}")
    except json.JSONDecodeError:
        return {}


def summarize_reports(out_dir: Path, total_configs: int) -> dict[str, Any]:
    filter_rows = _read_tsv(out_dir / "filter.tsv")
    perf_rows = _read_tsv(out_dir / "perf.tsv")

    filter_by_stage: Counter[tuple[str, str]] = Counter((row["stage"], row["verdict"]) for row in filter_rows)
    filter_reasons: Counter[tuple[str, str, str]] = Counter((row["stage"], row["verdict"], row["reason"]) for row in filter_rows)
    filter_violations: Counter[str] = Counter()
    filter_advisories: Counter[str] = Counter()
    for row in filter_rows:
        details = _json_cell(row, "details")
        for violation in details.get("violations", []) or []:
            filter_violations[str(violation.get("reason", ""))] += 1
        for advisory in details.get("advisories", []) or []:
            filter_advisories[str(advisory.get("reason", ""))] += 1

    perf_statuses: Counter[str] = Counter(row["status"] for row in perf_rows)
    best_latency = None
    best_index = None
    best_config = None
    for row in perf_rows:
        if row.get("status") != "ok" or not row.get("latency_ms"):
            continue
        latency = float(row["latency_ms"])
        if best_latency is None or latency < best_latency:
            best_latency = latency
            best_index = int(row["index"])
            best_config = _json_cell(row, "config")

    pre_seen = {int(row["index"]) for row in filter_rows if row["stage"] == "pre_compile"}
    post_seen = {int(row["index"]) for row in filter_rows if row["stage"] == "post_compile"}
    pre_reject = {int(row["index"]) for row in filter_rows if row["stage"] == "pre_compile" and row["verdict"] == "reject"}
    post_reject = {int(row["index"]) for row in filter_rows if row["stage"] == "post_compile" and row["verdict"] == "reject"}
    benchmark_seen = {int(row["index"]) for row in perf_rows}
    benchmark_ok = {int(row["index"]) for row in perf_rows if row["status"] == "ok"}
    compile_or_lower_failed = (set(range(total_configs)) - pre_seen) | ((pre_seen - pre_reject) - post_seen)
    kept_without_benchmark = (post_seen - post_reject) - benchmark_seen

    return {
        "total_configs": total_configs,
        "stage_reductions": {
            "pre_compile_seen": len(pre_seen),
            "pre_compile_filtered": len(pre_reject),
            "remaining_after_pre_compile": total_configs - len(pre_reject),
            "post_compile_seen": len(post_seen),
            "post_compile_filtered": len(post_reject),
            "remaining_after_post_compile": total_configs - len(pre_reject) - len(post_reject),
            "benchmarked": len(benchmark_seen),
            "benchmark_ok": len(benchmark_ok),
            "compile_or_lower_failed": len(compile_or_lower_failed),
        },
        "pre_compile_keep": filter_by_stage[("pre_compile", "keep")],
        "pre_compile_reject": filter_by_stage[("pre_compile", "reject")],
        "post_compile_keep": filter_by_stage[("post_compile", "keep")],
        "post_compile_reject": filter_by_stage[("post_compile", "reject")],
        "benchmark_rows": len(perf_rows),
        "benchmark_ok": perf_statuses["ok"],
        "benchmark_error": perf_statuses["error"],
        "benchmark_timeout": perf_statuses["timeout"],
        "compile_or_lower_failed": len(compile_or_lower_failed),
        "kept_without_benchmark": len(kept_without_benchmark),
        "best_index": best_index,
        "best_latency_ms": best_latency,
        "best_config": best_config,
        "filter_stage_verdicts": dict(filter_by_stage),
        "filter_reasons": dict(filter_reasons),
        "filter_violations": dict(filter_violations),
        "filter_advisories": dict(filter_advisories),
        "benchmark_statuses": dict(perf_statuses),
    }


def _jsonable_summary(summary: dict[str, Any]) -> dict[str, Any]:
    jsonable: dict[str, Any] = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            jsonable[key] = {str(k): v for k, v in value.items()}
        else:
            jsonable[key] = value
    return jsonable


def write_summary(out_dir: Path, family: str, summary: dict[str, Any]) -> None:
    jsonable = _jsonable_summary(summary)
    (out_dir / "summary.json").write_text(json.dumps(jsonable, indent=2, sort_keys=True) + "\n")
    stage = summary["stage_reductions"]
    print(f"[{family}] total_configs={summary['total_configs']}")
    print(f"[{family}] pre_compile keep={summary['pre_compile_keep']} reject={summary['pre_compile_reject']}")
    print(f"[{family}] post_compile keep={summary['post_compile_keep']} reject={summary['post_compile_reject']}")
    print(f"[{family}] benchmark ok={summary['benchmark_ok']} error={summary['benchmark_error']} timeout={summary['benchmark_timeout']}")
    print(
        f"[{family}] stage_reductions="
        f"pre={stage['pre_compile_filtered']} "
        f"post={stage['post_compile_filtered']} "
        f"benchmarked={stage['benchmarked']}"
    )
    print(f"[{family}] compile_or_lower_failed={summary['compile_or_lower_failed']}")
    print(f"[{family}] kept_without_benchmark={summary['kept_without_benchmark']}")
    print(f"[{family}] best_index={summary['best_index']}")
    print(f"[{family}] best_latency_ms={summary['best_latency_ms']}")
    print(f"[{family}] best_config={summary['best_config']}")
    print(f"[{family}] filter_reasons={_format_counter(Counter(summary['filter_reasons']))}")
    print(f"[{family}] filter_violations={_format_counter(Counter(summary['filter_violations']))}")
    print(f"[{family}] filter_advisories={_format_counter(Counter(summary['filter_advisories']))}")
    if summary.get("tuner_error"):
        print(f"[{family}] tuner_error={summary['tuner_error']}")
    print(f"[{family}] out_dir={out_dir.resolve()}")


def _maybe_limit_configs(configs: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.max_configs is None:
        return configs
    return configs[: args.max_configs]


def _run_tuner(
    tuner: AutoTuner,
    configs: list[dict[str, Any]],
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    try:
        result = tuner.run(
            warmup=args.warmup,
            rep=args.rep,
            timeout=args.timeout,
            enable_grouped_compile=args.grouped,
            group_compile_size=args.group_compile_size,
        )
    except Exception as exc:
        summary = summarize_reports(out_dir, len(configs))
        summary["tuner_error"] = f"{type(exc).__name__}: {exc}"
        return summary

    summary = summarize_reports(out_dir, len(configs))
    summary["tuner_best_config"] = result.config
    summary["tuner_best_latency_ms"] = result.latency
    return summary


def run_dense(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    configs = _maybe_limit_configs(dense_gemm_configs(), args)
    kernel = make_dense_gemm_kernel(args.m, args.n, args.k, T.bfloat16)
    tuner = (
        AutoTuner.from_kernel(kernel=kernel, configs=configs)
        .set_compile_args(out_idx=[-1], target="cuda", execution_backend="tvm_ffi")
        .set_profile_args(supply_type=tilelang.TensorSupplyType.Integer, skip_check=True)
        .set_filter_args(
            True,
            action=args.filter_action,
            report_path=str(out_dir / "filter.tsv"),
            kernel_type="dense_gemm",
        )
        .set_benchmark_report_path(out_dir / "perf.tsv")
    )
    return _run_tuner(tuner, configs, args, out_dir)


def _decorated_tuner(
    autotune_impl,
    call_args: tuple[Any, ...],
    call_kwargs: dict[str, Any],
    configs: list[dict[str, Any]],
    args: argparse.Namespace,
    out_dir: Path,
    kernel_type: str,
):
    autotune_impl.warmup = args.warmup
    autotune_impl.rep = args.rep
    autotune_impl.timeout = args.timeout
    autotune_impl.supply_type = tilelang.TensorSupplyType.Integer
    autotune_impl.skip_check = True
    autotune_impl.manual_check_prog = lambda *unused_args, **unused_kwargs: None
    autotune_impl.filter = {
        "enabled": True,
        "action": args.filter_action,
        "report_path": str(out_dir / "filter.tsv"),
        "kernel_type": kernel_type,
    }

    mode = autotune_impl.jit_impl.initialize_jit_mode(*call_args, **call_kwargs)
    tuner = autotune_impl.get_tunner()
    tuner.run = AutoTuner.run.__get__(tuner, AutoTuner)
    tuner.fn = autotune_impl.jit_impl.func.orig_func
    tuner.configs = configs
    tuner.set_benchmark_report_path(out_dir / "perf.tsv")
    tuner.jit_compile = autotune_impl._make_jit_compile_func(mode, call_args, call_kwargs)

    def jit_elaborate(**config_arg):
        config_arg.pop(_PASS_CONFIGS_KEY, None)
        merged = dict(call_kwargs)
        merged.update(config_arg)
        return autotune_impl.jit_impl.get_tir(*call_args, **merged)

    tuner.jit_elaborate = jit_elaborate
    tuner.set_kernel_parameters(
        (
            _normalize_value(call_args, sort_dict_items=True),
            _normalize_value(call_kwargs, sort_dict_items=True),
        ),
        autotune_impl.jit_impl.signature.parameters,
    )
    return tuner


def run_quant_fp4(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    _add_example_dir("dequantize_gemm")
    import example_dequant_gemm_bf16_fp4_hopper as quant_fp4

    configs = _maybe_limit_configs(quant_fp4.get_configs(), args)
    tuner = _decorated_tuner(
        quant_fp4.matmul,
        (args.m, args.n, args.k, T.bfloat16, T.bfloat16, T.float32),
        {"num_bits": 4, "fast_dequant": args.fast_dequant},
        configs,
        args,
        out_dir,
        "quantized_gemm",
    )
    return _run_tuner(tuner, configs, args, out_dir)


def run_quant_w4a8(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    _add_example_dir("dequantize_gemm")
    import example_dequant_gemm_w4a8 as quant_w4a8

    configs = _maybe_limit_configs(quant_w4a8.get_configs(), args)
    tuner = _decorated_tuner(
        quant_w4a8.matmul_int8xint4,
        (args.m, args.n, args.k, T.int8, T.int32, T.int32),
        {"num_bits": 4},
        configs,
        args,
        out_dir,
        "quantized_gemm",
    )
    return _run_tuner(tuner, configs, args, out_dir)


def run_blocksparse(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    _add_example_dir("blocksparse_gemm")
    import example_blocksparse_gemm as blocksparse

    configs = _maybe_limit_configs(blocksparse.get_configs(), args)
    tuner = _decorated_tuner(
        blocksparse.blocksparse_matmul,
        (),
        {"M": args.m, "N": args.n, "K": args.k},
        configs,
        args,
        out_dir,
        "sparse_gemm",
    )
    return _run_tuner(tuner, configs, args, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GEMM-family autotune filtering experiments.")
    parser.add_argument("--family", choices=["dense", "quant_w4a8", "quant_fp4", "blocksparse", "all"], default="all")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--out-root", type=Path, default=Path("/tmp/tilelang_gemm_family_filter"))
    parser.add_argument("--filter-action", choices=["reject", "report"], default="reject")
    parser.add_argument("--grouped", action="store_true")
    parser.add_argument("--group-compile-size", type=int, default=8)
    parser.add_argument("--fast-dequant", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-configs", type=int, default=None, help="Debug only: cap each family after config generation.")
    parser.add_argument("--disable-cache", action="store_true", help="Disable TileLang cache for this benchmark process.")
    args = parser.parse_args()

    if args.disable_cache:
        tilelang.disable_cache()

    selected = ["dense", "quant_w4a8", "quant_fp4", "blocksparse"] if args.family == "all" else [args.family]
    runners = {
        "dense": run_dense,
        "quant_w4a8": run_quant_w4a8,
        "quant_fp4": run_quant_fp4,
        "blocksparse": run_blocksparse,
    }

    args.out_root.mkdir(parents=True, exist_ok=True)
    all_summaries: dict[str, Any] = {}
    start = time.perf_counter()
    for family in selected:
        family_dir = args.out_root / family
        family_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== {family} ===")
        family_start = time.perf_counter()
        summary = runners[family](args, family_dir)
        summary["elapsed_s"] = time.perf_counter() - family_start
        write_summary(family_dir, family, summary)
        all_summaries[family] = summary

    root_summary = {family: _jsonable_summary(summary) for family, summary in all_summaries.items()}
    (args.out_root / "summary.json").write_text(json.dumps(root_summary, indent=2, sort_keys=True) + "\n")
    print(f"total_elapsed_s={time.perf_counter() - start:.3f}")
    print(f"summary={args.out_root.resolve() / 'summary.json'}")


if __name__ == "__main__":
    main()
