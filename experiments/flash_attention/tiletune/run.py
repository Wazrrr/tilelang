"""Compare exhaustive FlashAttention search and TileTune at a fixed top-k."""

import argparse
from collections import Counter
import csv
import json
import os
import random
import statistics
import subprocess
import sys
import time

from experiments._common import observe_compilation, device_info, positive_int, prepare_run, select_configs, source_hashes, write_json
from experiments._tiletune import add_arguments, rank_info, winner_summary


METHODS = ("brute_force", "tiletune")


def top_k(value):
    return None if value == "all" else positive_int(value)


def workload_flops(args):
    """Useful QK and AV FLOPs; causal attention counts the lower triangle."""
    pairs = args.sequence * (args.sequence + 1) // 2 if args.causal else args.sequence**2
    return 4 * args.batch * args.heads * args.dim * pairs


def benchmark(args, configs, inputs, expected, profile, target):
    """Use the same autotuner/validation path, retaining failed compile attempts."""
    from tilelang.autotuner import AutoTuner
    from experiments.flash_attention.kernel import make_kernel, check_accuracy, PASS_CONFIGS

    outcomes = {}

    class ObservedTuner(AutoTuner):
        def _prepare_compile_execution(self, *a, **kw):
            return observe_compilation(super()._prepare_compile_execution(*a, **kw), outcomes)

    tuner = (
        ObservedTuner(make_kernel(args.batch, args.heads, args.sequence, args.dim, args.causal), configs)
        .set_compile_args(target=target, execution_backend="tvm_ffi", out_idx=[3], pass_configs=PASS_CONFIGS)
        .set_profile_args(
            supply_prog=lambda params: inputs,
            ref_prog=lambda *unused: expected,
            manual_check_prog=check_accuracy,
            backend=args.backend,
            cache_input_tensors=False,  # The supply function returns the same seeded tensors.
            rtol=0.01,
            atol=0.01,
            max_mismatched_ratio=0.0,
        )
        .set_benchmark_report_path(args.output / "benchmarks.tsv")
    )
    if args.method == "tiletune":
        tuner.set_tiletune_args(
            True,
            mode="report_only",
            top_k=args.top_k,
            ranking_metric="pipeline_time",
            performance_model=profile,
            report_path=str(args.output / "tiletune.json"),
            attention_spill_budget_registers_per_thread=args.spill_budget_registers_per_thread,
            max_spill_bytes=None,
            max_local_bytes=None,
        )
    failure = None
    started = time.perf_counter()
    try:
        result = tuner.run(
            warmup=args.warmup,
            rep=args.rep,
            timeout=args.timeout,
            early_stop=False,
            use_pipeline=False,
            enable_grouped_compile=args.group_size > 1,
            group_compile_size=args.group_size,
            benchmark_multi_gpu=False,
        )
    except RuntimeError as error:
        if not str(error).startswith(("Auto-tuning failed:", "TileTune top_k selected no eligible")):
            raise
        result, failure = None, str(error)
    elapsed = time.perf_counter() - started
    if (args.output / "benchmarks.tsv").exists():
        with (args.output / "benchmarks.tsv").open() as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                outcomes[int(row["index"])] = dict(
                    status="benchmarked" if row["status"] == "ok" else "benchmark_" + row["status"],
                    latency_ms=float(row["latency_ms"]) if row["latency_ms"] else None,
                    error=row["error"] or None,
                )
    return result, tuner.tiletune_report, outcomes, elapsed, failure


def run_one(args):
    import torch
    from tilelang.cache.kernel_cache import KernelCache
    from tilelang.tiletune import profile_device
    from tilelang.tiletune.profiling.device_profile import current_target
    from experiments.flash_attention.kernel import get_configs, make_inputs, reference

    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    target = current_target()
    grid = get_configs()
    original_indices, configs = select_configs(grid, args.config_indices)
    inputs = make_inputs(args.batch, args.heads, args.sequence, args.dim, args.seed)
    expected = reference(*inputs, args.causal)
    torch.cuda.synchronize()
    profile, profile_seconds = None, 0.0
    if args.method == "tiletune":
        started = time.perf_counter()
        profile = profile_device(input_dtype="float16", cache_path=args.device_profile, memory_regime=args.memory_regime)
        profile_seconds = time.perf_counter() - started
    write_json(
        args.output / "experiment.json",
        dict(
            method=args.method,
            arguments=vars(args),
            configs=configs,
            original_indices=original_indices,
            original_grid_size=len(grid),
            target=target,
            devices=device_info([0]),
            profile=profile,
            profile_seconds=profile_seconds,
            source_sha256=source_hashes("examples/flash_attention/example_mha_fwd_bshd.py"),
            native_build=KernelCache._get_tilelang_lib_stamp(),
            cold_kernel_cache=True,
            cold_autotune_cache=True,
        ),
    )
    report, selection_seconds = None, 0.0
    selected = list(range(len(configs)))
    print(f"{args.method}: grid={len(configs)}, top_k={args.top_k}", flush=True)
    result, tiletune_report, outcomes, tuning_seconds, failure = benchmark(args, configs, inputs, expected, profile, target)
    if tiletune_report is not None:
        report = tiletune_report
        if report["selection"] is not None:
            selected = report["selection"]["selected_indices"]
            selection_seconds = report["selection"]["wall_time_ms"] / 1000
        # TileTune's run includes its pre-compilation analysis/selection phase.
        tuning_seconds -= selection_seconds
    records = report["configs"] if report else [dict(index=i, config=cfg, selected=True) for i, cfg in enumerate(configs)]
    for record in records:
        record["original_index"] = original_indices[record["index"]]
        record["selected"] = record["index"] in selected
        record.setdefault("status", "not_attempted")
        record.update(outcomes.get(record["index"], {}))
    if report is not None:
        write_json(args.output / f"{args.method}.json", report)
    write_json(args.output / "outcomes.json", records)
    winner = None
    if result is not None:
        index = configs.index(result.config)
        winner = dict(
            index=index,
            original_index=original_indices[index],
            config=result.config,
            latency_ms=result.latency,
            tflops=workload_flops(args) / (result.latency * 1e9),
        )
        if report:
            winner.update(rank_info(report, index))
        if args.method == "tiletune":
            winner.update(winner_summary(report, result.config, result.latency))
    summary = dict(
        method=args.method,
        status="ok" if result is not None else "failed",
        error=failure,
        grid_size=len(configs),
        requested_k=None if args.method == "brute_force" else args.top_k,
        selected_count=len(selected),
        successful_count=sum(r["status"] == "benchmarked" for r in records),
        selected_failed_count=sum(r["index"] in selected and r["status"] != "benchmarked" for r in records),
        statuses=dict(Counter(r["status"] for r in records)),
        winner=winner,
        profile_seconds=profile_seconds,
        selection_seconds=selection_seconds,
        compile_and_benchmark_seconds=tuning_seconds,
        tuning_seconds=selection_seconds + tuning_seconds,
    )
    timings = []
    if (args.output / "timings.tsv").exists():
        with (args.output / "timings.tsv").open() as stream:
            timings = list(csv.DictReader(stream, delimiter="\t"))
    summary["compile_work_seconds"] = (
        sum(
            float(row["duration_ms"])
            for row in timings
            if row["stage"] in ("autotune.compile_unit.single", "autotune.compile_unit.grouped")
        )
        / 1000
    )
    summary["benchmark_work_seconds"] = (
        sum(float(row["duration_ms"]) for row in timings if row["stage"] == "autotune.benchmark_call_wall") / 1000
    )
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return summary


def remeasure_winners(args, summaries):
    """Remeasure only the already chosen winners; this is separate validation work."""
    import tilelang
    import torch
    from experiments.flash_attention.kernel import make_kernel, make_inputs, reference, check_accuracy, PASS_CONFIGS
    from tilelang.tiletune.profiling.device_profile import current_target

    os.environ["TILELANG_AUTOTUNE_TIMING_LOG"] = str(args.output / "validation_timings.tsv")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = make_inputs(args.batch, args.heads, args.sequence, args.dim, args.seed)
    expected = reference(*inputs, args.causal)
    torch.cuda.synchronize()
    factory = make_kernel(args.batch, args.heads, args.sequence, args.dim, args.causal)
    kernels = {}
    for summary in summaries:
        if summary.get("winner"):
            cfg = summary["winner"]["config"]
            key = json.dumps(cfg, sort_keys=True)
            if key not in kernels:
                kernel = tilelang.compile(
                    factory(**cfg), target=current_target(), out_idx=[3], pass_configs=PASS_CONFIGS, execution_backend="tvm_ffi"
                )
                profiler = kernel.get_profiler()
                profiler.manual_assert_close(lambda *unused: expected, input_tensors=inputs, manual_check_prog=check_accuracy)
                kernels[key] = profiler
    samples = {key: [] for key in kernels}
    randomizer = random.Random(args.seed)
    for _ in range(args.validation_repeats):
        order = list(kernels)
        randomizer.shuffle(order)
        for key in order:
            samples[key].append(kernels[key].do_bench(input_tensors=inputs, backend=args.backend, warmup=args.warmup, rep=args.rep))
    for summary in summaries:
        if summary.get("winner"):
            values = samples[json.dumps(summary["winner"]["config"], sort_keys=True)]
            summary["validation_latency_ms"] = statistics.median(values)
            summary["validation_samples_ms"] = values
            summary["validation_tflops"] = workload_flops(args) / (summary["validation_latency_ms"] * 1e9)


def compare_results(summaries, method_outputs):
    """Attach common-grid evaluation without changing a method's frozen selection."""
    brute = next(row for row in summaries if row["method"] == "brute_force")
    if not brute.get("winner"):
        return
    records = json.loads((method_outputs["brute_force"] / "outcomes.json").read_text())
    oracle_times = {r["index"]: r["latency_ms"] for r in records if r["status"] == "benchmarked"}
    for row in summaries:
        if row.get("winner"):
            row["validated_performance_vs_brute_force"] = brute["validation_latency_ms"] / row["validation_latency_ms"]
            row["tuning_speedup_vs_brute_force"] = brute["tuning_seconds"] / row["tuning_seconds"]
        if row["method"] == "brute_force" or not (method_outputs[row["method"]] / f"{row['method']}.json").exists():
            continue
        report = json.loads((method_outputs[row["method"]] / f"{row['method']}.json").read_text())
        row["brute_force_winner_model_rank"] = rank_info(report, brute["winner"]["index"])
        selected = [record["index"] for record in report["configs"] if record["selected"]]
        valid = [idx for idx in selected if idx in oracle_times]
        row["selected_candidates_with_oracle_measurement"] = len(valid)
        row["top_k_oracle_retained_performance"] = min(oracle_times.values()) / min(oracle_times[idx] for idx in valid) if valid else None


def run_all(args):
    summaries, outputs = [], {}
    # Model selections are complete before the exhaustive baseline produces labels.
    for method in ("tiletune", "brute_force"):
        output = outputs[method] = args.output / method
        command = [
            sys.executable,
            "-m",
            "experiments.flash_attention.tiletune.run",
            "--method",
            method,
            "--output",
            str(args.output),
            "--run-name",
            method,
        ]
        for name in (
            "batch",
            "heads",
            "sequence",
            "dim",
            "spill_budget_registers_per_thread",
            "top_k",
            "workers",
            "warmup",
            "rep",
            "timeout",
            "seed",
            "backend",
            "memory_regime",
            "group_size",
            "validation_repeats",
        ):
            value = getattr(args, name)
            command += ["--" + name.replace("_", "-"), "all" if name == "top_k" and value is None else str(value)]
        if args.device_profile is not None:
            command += ["--device-profile", str(args.device_profile)]
        if args.config_indices is not None:
            command += ["--config-indices", *map(str, args.config_indices)]
        if args.causal:
            command.append("--causal")
        log_path = args.output / f"{method}.log"
        print(f"Running {method}; log: {log_path}", flush=True)
        with log_path.open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if (output / "summary.json").exists():
            summary = json.loads((output / "summary.json").read_text())
        else:
            summary = dict(method=method, status="failed", error=f"Process exited {completed.returncode}; see {log_path}")
        summaries.append(summary)
        print(f"{method}: {summary['status']}", flush=True)
    summaries.sort(key=lambda row: METHODS.index(row["method"]))
    started = time.perf_counter()
    remeasure_winners(args, summaries)
    validation_seconds = time.perf_counter() - started
    compare_results(summaries, outputs)
    write_json(args.output / "comparison.json", dict(methods=summaries, validation_seconds=validation_seconds, top_k=args.top_k))
    print("\nMethod        Selected  Success  Winner (ms)  Tuning (s)  Oracle@K")
    for row in summaries:
        if not row.get("winner"):
            print(f"{row['method']:12} FAILED: {row.get('error')}")
            continue
        retained = row.get("top_k_oracle_retained_performance", 1.0)
        retained_text = f"{retained:.2%}" if retained is not None else "unavailable"
        print(
            f"{row['method']:12} {row['selected_count']:8} {row['successful_count']:8} "
            f"{row['validation_latency_ms']:12.6f} {row['tuning_seconds']:11.3f} {retained_text:>9}"
        )
        if row.get("brute_force_winner_model_rank"):
            print(
                f"  Selected winner's model rank: {row['winner'].get('predicted_rank')} (original config #{row['winner']['original_index']})"
            )
            print(f"  Brute-force winner's model rank: {row['brute_force_winner_model_rank']}")
    print(f"Results: {args.output}")
    return all(row["status"] == "ok" for row in summaries)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, "flash_attention")
    parser.add_argument("--batch", type=positive_int, default=1)
    parser.add_argument("--heads", type=positive_int, default=16)
    parser.add_argument("--sequence", type=positive_int, default=4096)
    parser.add_argument("--dim", type=positive_int, default=128)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--spill-budget-registers-per-thread", type=int, default=32)
    parser.add_argument("--method", choices=[*METHODS, "all"], default="tiletune")
    parser.add_argument("--top-k", type=top_k, default=20, help="Candidate budget; all preserves exhaustive TileTune ranking")
    parser.add_argument("--backend", choices=["cudagraph", "event"], default="cudagraph")
    parser.add_argument("--validation-repeats", type=positive_int, default=5)
    args = parser.parse_args(argv)
    if args.spill_budget_registers_per_thread < 0:
        parser.error("--spill-budget-registers-per-thread must be nonnegative")
    return args


def main():
    args = parse_args()
    prepare_run(args)
    successful = run_all(args) if args.method == "all" else run_one(args)["status"] == "ok"
    if not successful:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
