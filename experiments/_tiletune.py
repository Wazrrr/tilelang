"""Run exhaustive tuning and report the measured winner's pipeline-time rank."""

from collections import Counter
from pathlib import Path
import time

from experiments._common import add_run_arguments, device_info, positive_int, select_configs, source_hashes, write_json


def add_arguments(parser, family):
    add_run_arguments(parser, f"experiments/results/{family}/tiletune")
    parser.add_argument("--device-profile", type=Path, help="Reusable primitive profile; measure missing rates before tuning")
    parser.add_argument("--memory-regime", choices=["streaming", "cached"], default="streaming")
    parser.add_argument("--group-size", type=positive_int, default=1, help="Configurations per compilation unit; 1 disables grouping")


def winner_summary(report, winner_config, latency_ms):
    """Look up the measured winner in the model ranking, retaining unknowns and ties."""
    record = next(record for record in report["configs"] if record["config"] == winner_config)
    entry = next(entry for entry in report["ranking"] if entry["index"] == record["index"])
    scored = entry["tier"] == "eligible" and entry["score"] is not None
    return {
        "config": winner_config,
        "index": record["index"],
        "latency_ms": latency_ms,
        "predicted_rank": entry["rank"] if scored else None,
        "tie_first_rank": entry["tie_first_rank"] if scored else None,
        "tie_last_rank": entry["tie_last_rank"] if scored else None,
        "report_position": entry["rank"],
        "tier": entry["tier"],
        "score_cycles": entry["score"],
        "scored_candidates": sum(row["tier"] == "eligible" and row["score"] is not None for row in report["ranking"]),
        "total_candidates": len(report["configs"]),
    }


def run(args, *, kernel, grid, inputs, reference, dtype, out_idx, workload, kernel_source, pass_configs=None, check=None, options=None):
    import torch
    from tilelang.autotuner import AutoTuner
    from tilelang.tiletune import profile_device
    from tilelang.tiletune.profiling.device_profile import current_target

    indices, configs = select_configs(grid, args.config_indices)
    target = current_target()
    print(f"Preparing {dtype} device profile (cached rates are reused; missing rates require GPU profiling)", flush=True)
    started = time.perf_counter()
    profile = profile_device(input_dtype=dtype, cache_path=args.device_profile, memory_regime=args.memory_regime)
    profile_seconds = time.perf_counter() - started
    # Primitive rates are fixed before candidate measurements. They are never fitted to this grid.
    write_json(
        args.output / "experiment.json",
        dict(
            workload=workload,
            arguments=vars(args),
            target=target,
            devices=device_info([torch.cuda.current_device()]),
            profile=profile,
            profile_seconds=profile_seconds,
            ranking_metric="pipeline_time",
            mode="report_only",
            original_grid_size=len(grid),
            original_indices=indices,
            configs=configs,
            source_sha256=source_hashes(kernel_source),
        ),
    )
    tuner = (
        AutoTuner(kernel, configs)
        .set_compile_args(target=target, execution_backend="tvm_ffi", out_idx=out_idx, pass_configs=pass_configs)
        .set_profile_args(
            supply_prog=lambda params: inputs,
            ref_prog=lambda *unused: reference,
            manual_check_prog=check,
            backend="cudagraph",
            cache_input_tensors=True,
            rtol=0.01,
            atol=0.01,
            max_mismatched_ratio=0.0,
        )
        .set_benchmark_report_path(str(args.output / "benchmarks.tsv"))
        .set_tiletune_args(
            True,
            mode="report_only",
            ranking_metric="pipeline_time",
            performance_model=profile,
            report_path=str(args.output / "tiletune.json"),
            **(options or {}),
        )
    )
    print(f"Ranking and tuning {len(configs)}/{len(grid)} configurations with pipeline_time", flush=True)
    started = time.perf_counter()
    result = tuner.run(
        warmup=args.warmup,
        rep=args.rep,
        timeout=args.timeout,
        early_stop=False,
        use_pipeline=False,
        enable_grouped_compile=args.group_size > 1,
        group_compile_size=args.group_size,
    )
    tuning_seconds = time.perf_counter() - started
    report = tuner.tiletune_report
    winner = winner_summary(report, result.config, result.latency)
    winner["original_index"] = indices[winner["index"]]
    summary = dict(
        workload=workload,
        ranking_metric="pipeline_time",
        profile_seconds=profile_seconds,
        tuning_seconds=tuning_seconds,
        statuses=dict(Counter(record["status"] for record in report["configs"])),
        winner=winner,
    )
    write_json(args.output / "summary.json", summary)
    print(f"Measured winner: {winner['latency_ms']:.6f} ms, original config #{winner['original_index']}")
    print(f"Winner config: {winner['config']}")
    if winner["predicted_rank"] is not None:
        print(
            f"Winner's pipeline_time rank: {winner['predicted_rank']}/{winner['total_candidates']} "
            f"(tie range {winner['tie_first_rank']}-{winner['tie_last_rank']}; "
            f"{winner['scored_candidates']} scored candidates)"
        )
    else:
        print(
            f"Winner's pipeline_time rank: unavailable (tier={winner['tier']}; "
            f"report position {winner['report_position']}/{winner['total_candidates']})"
        )
    print(f"Tuning: {tuning_seconds:.3f} s; device profiling: {profile_seconds:.3f} s")
    print(f"Results: {args.output}")
