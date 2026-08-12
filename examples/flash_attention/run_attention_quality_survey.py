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


def make_flashattn_kernel(
    batch: int,
    heads: int,
    seq_q: int,
    seq_kv: int,
    dim: int,
    is_causal: bool,
):
    def kernel(block_M=64, block_N=64, num_stages=1, threads=128):
        scale = (1.0 / dim) ** 0.5 * 1.44269504
        q_shape = [batch, heads, seq_q, dim]
        kv_shape = [batch, heads, seq_kv, dim]
        dtype = T.float16
        accum_dtype = T.float32
        past_len = seq_kv - seq_q

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            Output: T.Tensor(q_shape, dtype),
        ):
            with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([block_M, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)

                T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = (
                    T.min(T.ceildiv(seq_kv, block_N), T.ceildiv((bx + 1) * block_M + past_len, block_N))
                    if is_causal
                    else T.ceildiv(seq_kv, block_N)
                )

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(K[bz, by, k * block_N : (k + 1) * block_N, :], K_shared)
                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            q_idx = bx * block_M + i + past_len
                            k_idx = k * block_N + j
                            acc_s[i, j] = T.if_then_else(q_idx >= k_idx, 0, -T.infinity(acc_s.dtype))
                    else:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(k * block_N + j >= seq_kv, -T.infinity(acc_s.dtype), 0)
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    for i in T.Parallel(block_M):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)

                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] *= scores_scale[i]

                    T.copy(V[bz, by, k * block_N : (k + 1) * block_N, :], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[bz, by, bx * block_M : (bx + 1) * block_M, :])

        return main

    return kernel


def make_configs() -> list[dict]:
    return [
        {
            "block_M": block_m,
            "block_N": block_n,
            "num_stages": num_stages,
            "threads": threads,
        }
        for block_m, block_n, num_stages, threads in itertools.product(
            [64, 128, 256],
            [32, 64, 128, 256],
            [1, 2],
            [128, 256],
        )
    ]


def summarize_report(path: Path) -> tuple[int, Counter[str], Counter[str]]:
    rows = 0
    reasons: Counter[str] = Counter()
    advisories: Counter[str] = Counter()
    if not path.exists():
        return rows, reasons, advisories
    with path.open(newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            rows += 1
            try:
                details = json.loads(row.get("details", "") or "{}")
            except json.JSONDecodeError:
                details = {}
            for violation in details.get("violations", []) or []:
                reasons[str(violation.get("reason", ""))] += 1
            for advisory in details.get("advisories", []) or []:
                advisories[str(advisory.get("reason", ""))] += 1
    return rows, reasons, advisories


def _read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def _json_cell(row: dict[str, str], key: str) -> dict:
    try:
        return json.loads(row.get(key, "") or "{}")
    except json.JSONDecodeError:
        return {}


def summarize_filter_effect(out_dir: Path, total_configs: int) -> dict:
    resource_rows = _read_tsv(out_dir / "resource.tsv")
    quality_rows = _read_tsv(out_dir / "quality.tsv")
    perf_rows = _read_tsv(out_dir / "perf.tsv")

    resource_by_stage = Counter((row["stage"], row["verdict"]) for row in resource_rows)
    resource_reasons = Counter((row["stage"], row["verdict"], row["reason"]) for row in resource_rows)
    quality_by_verdict = Counter(row["verdict"] for row in quality_rows)
    quality_reasons = Counter(row["reason"] for row in quality_rows)
    quality_violations = Counter()
    quality_advisories = Counter()
    for row in quality_rows:
        details = _json_cell(row, "details")
        for violation in details.get("violations", []) or []:
            quality_violations[str(violation.get("reason", ""))] += 1
        for advisory in details.get("advisories", []) or []:
            quality_advisories[str(advisory.get("reason", ""))] += 1

    perf_statuses = Counter(row["status"] for row in perf_rows)
    pre_seen = {int(row["index"]) for row in resource_rows if row["stage"] == "pre_compile"}
    post_seen = {int(row["index"]) for row in resource_rows if row["stage"] == "post_compile"}
    pre_reject = {int(row["index"]) for row in resource_rows if row["stage"] == "pre_compile" and row["verdict"] == "reject"}
    post_reject = {int(row["index"]) for row in resource_rows if row["stage"] == "post_compile" and row["verdict"] == "reject"}
    quality_seen = {int(row["index"]) for row in quality_rows}
    quality_reject = {int(row["index"]) for row in quality_rows if row["verdict"] == "reject"}
    benchmark_seen = {int(row["index"]) for row in perf_rows}
    benchmark_ok = {int(row["index"]) for row in perf_rows if row["status"] == "ok"}
    compile_or_lower_failed = set(range(total_configs)) - pre_reject - post_reject - quality_seen

    return {
        "total_configs": total_configs,
        "stage_reductions": {
            "pre_compile_seen": len(pre_seen),
            "pre_compile_filtered": len(pre_reject),
            "remaining_after_pre_compile": total_configs - len(pre_reject),
            "post_compile_seen": len(post_seen),
            "post_compile_filtered": len(post_reject),
            "remaining_after_post_compile": total_configs - len(pre_reject) - len(post_reject),
            "quality_seen": len(quality_seen),
            "quality_filtered": len(quality_reject),
            "remaining_after_quality": total_configs - len(pre_reject) - len(post_reject) - len(quality_reject),
            "benchmarked": len(benchmark_seen),
            "benchmark_ok": len(benchmark_ok),
            "compile_or_lower_failed": len(compile_or_lower_failed),
        },
        "resource_stage_verdicts": {str(key): value for key, value in resource_by_stage.items()},
        "resource_reasons": {str(key): value for key, value in resource_reasons.items()},
        "quality_verdicts": dict(quality_by_verdict),
        "quality_reasons": dict(quality_reasons),
        "quality_violations": dict(quality_violations),
        "quality_advisories": dict(quality_advisories),
        "benchmark_statuses": dict(perf_statuses),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Survey exact post-compile quality targets for flash attention.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--seq-q", type=int, default=1024)
    parser.add_argument("--seq-kv", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rep", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--group-compile-size", type=int, default=8)
    parser.add_argument("--quality-action", choices=["reject", "report"], default="report")
    parser.add_argument("--max-configs", type=int, default=None, help="Debug only: cap the config space after generation.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs = make_configs()
    if args.max_configs is not None:
        configs = configs[: args.max_configs]
    print(f"config_count={len(configs)}")

    kernel = make_flashattn_kernel(args.batch, args.heads, args.seq_q, args.seq_kv, args.dim, args.causal)
    tuner = (
        AutoTuner.from_kernel(kernel=kernel, configs=configs)
        .set_compile_args(
            out_idx=[-1],
            target="cuda",
            execution_backend="tvm_ffi",
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        )
        .set_profile_args(supply_type=tilelang.TensorSupplyType.Integer, skip_check=True)
        .set_resource_filter_args(True, report_path=str(args.out_dir / "resource.tsv"), kernel_type="attention")
        .set_quality_filter_args(
            True,
            action=args.quality_action,
            report_path=str(args.out_dir / "quality.tsv"),
            kernel_type="attention",
        )
        .set_benchmark_report_path(args.out_dir / "perf.tsv")
    )

    start = time.perf_counter()
    result = None
    tuner_error = None
    try:
        result = tuner.run(
            warmup=args.warmup,
            rep=args.rep,
            enable_grouped_compile=True,
            group_compile_size=args.group_compile_size,
        )
    except Exception as exc:
        tuner_error = f"{type(exc).__name__}: {exc}"
    rows, violations, advisories = summarize_report(args.out_dir / "quality.tsv")
    summary = summarize_filter_effect(args.out_dir, len(configs))
    summary["best_config"] = None if result is None else result.config
    summary["best_latency_ms"] = None if result is None else result.latency
    summary["elapsed_s"] = time.perf_counter() - start
    if tuner_error is not None:
        summary["tuner_error"] = tuner_error
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"best_config={summary['best_config']}")
    print(f"best_latency_ms={summary['best_latency_ms']}")
    print(f"elapsed_s={summary['elapsed_s']:.3f}")
    print(f"quality_rows={rows}")
    print(f"quality_violations={dict(sorted(violations.items()))}")
    print(f"quality_advisories={dict(sorted(advisories.items()))}")
    print(f"stage_reductions={summary['stage_reductions']}")
    if tuner_error is not None:
        print(f"tuner_error={tuner_error}")
    print(f"summary={args.out_dir.resolve() / 'summary.json'}")
    print(f"out_dir={args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
