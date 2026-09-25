"""System ablations on the same example kernels and pools as TileTune."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

from experiments.utils.io import write_json
from experiments.common.spec import Device, Workload, support_reason
from experiments.families import FAMILIES, family_module

VARIANTS = {
    "baseline": (False, False, False),
    "pipeline": (True, False, False),
    "grouped": (False, True, False),
    "multi_gpu": (False, False, True),
    "combined": (True, True, True),
}
OPS = {family: op for op, family in FAMILIES.items()}


def system_plan(family, *, workloads=None, variants=None, indices=None):
    cases = family_module(OPS[family], "cases").cases(holdout=True)
    if workloads:
        if set(workloads) - {w.name for w in cases}:
            raise ValueError("unknown final workload name for this family")
        cases = [w for w in cases if w.name in workloads]
    from experiments.utils.cli import select_configs

    pool = family_module(OPS[family], "spaces").get_configs()
    return [
        dict(
            workload=w.to_dict(),
            variant=v,
            indices=select_configs(pool, indices)[0],
        )
        for w in cases
        for v in (variants or VARIANTS)
    ]


def worker(request_path, output):
    import torch
    from tilelang.autotuner import AutoTuner, set_autotune_inputs
    from tilelang.tiletune import TileTuneConfig, current_target
    from experiments.utils.cli import device_info, source_hashes, select_configs, observe_compilation
    from .kernels import make_case

    request = json.loads(Path(request_path).read_text())
    w, settings = Workload(**request["workload"]), request["settings"]
    pipeline, grouped, multi_gpu = VARIANTS[request["variant"]]
    devices = list(range(torch.cuda.device_count())) if multi_gpu else [0]
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    target = current_target()
    compiler_target = target.compiler_target() if hasattr(target, "compiler_target") else {"kind": "cuda", "arch": "sm_80"}
    reason = support_reason(w, Device(name="ampere", target=compiler_target))
    if reason:
        write_json(
            output / "summary.json",
            dict(status="unsupported", workload=w.name, variant=request["variant"], reason=reason),
        )
        return
    case = make_case(w)
    pool = family_module(w.op, "spaces").get_configs()
    indices, configs = select_configs(pool, request["indices"])
    inputs = case.inputs("cuda", torch.Generator(device="cuda").manual_seed(settings["seed"]))
    case.check_input_values(inputs)
    references = {d: case.reference(*(value.to(f"cuda:{d}") for value in inputs)) for d in devices}
    for d in devices:
        torch.cuda.synchronize(d)
    outcomes = {}

    class ObservedTuner(AutoTuner):
        def _prepare_compile_execution(self, *args, **kwargs):
            return observe_compilation(super()._prepare_compile_execution(*args, **kwargs), outcomes)

    with set_autotune_inputs(inputs):
        tuner = (
            ObservedTuner(case.build, configs)
            .set_compile_args(out_idx=case.out_idx, pass_configs=case.pass_configs, target=target, execution_backend="tvm_ffi")
            .set_profile_args(
                ref_prog=lambda *values: references[values[0].device.index],
                manual_check_prog=case.check,
                backend="event",
                cache_input_tensors=False,
            )
            .set_benchmark_report_path(str(output / "benchmarks.tsv"))
        )
        if settings.get("tiletune"):
            tuner = tuner.set_tiletune_args(
                TileTuneConfig(
                    enabled=True,
                    ranking_metric=settings.get("metric", "memory"),
                    alpha=settings.get("alpha"),
                    mode="reject",
                    max_spill_bytes=settings.get("max_spill_bytes", 0),
                    max_local_bytes=settings.get("max_local_bytes"),
                    input_values=case.input_values or None,
                    report_path=str(output / "tiletune.json"),
                )
            )
    write_json(
        output / "experiment.json",
        dict(
            **request,
            configs=configs,
            original_indices=indices,
            target=target,
            source_sha256=source_hashes("experiments/common/system.py"),
            devices=device_info(devices),
            cold_kernel_cache=True,
            cold_autotune_cache=True,
        ),
    )
    started = time.perf_counter()
    result = tuner.run(
        warmup=settings["warmup"],
        rep=settings["rep"],
        timeout=settings["timeout"],
        early_stop=False,
        use_pipeline=pipeline,
        enable_grouped_compile=grouped,
        group_compile_size=settings["group_size"],
        benchmark_multi_gpu=multi_gpu,
        benchmark_devices=devices,
    )
    write_json(output / "compilation.json", outcomes)
    write_json(
        output / "summary.json",
        dict(
            status="completed",
            workload=w.name,
            variant=request["variant"],
            tuning_seconds=time.perf_counter() - started,
            config_count=len(configs),
            winner_config=result.config,
            winner_latency_ms=result.latency,
        ),
    )
    # Oracle-style per-config outcomes so this run can feed E3 verification.
    import csv as _csv
    from experiments.common.spaces import config_id
    from experiments.utils.results import TERMINAL

    records = []
    for idx, cfg in enumerate(configs):
        comp = outcomes.get(idx) or {}
        records.append(
            dict(
                index=idx,
                original_index=indices[idx],
                config=cfg,
                config_id=config_id(cfg),
                status="compilation_failed" if comp.get("status") == "compilation_failed" else "compiled",
                error=comp.get("error"),
                latency_ms=None,
            )
        )
    if (output / "benchmarks.tsv").exists():
        with (output / "benchmarks.tsv").open() as stream:
            for row in _csv.DictReader(stream, delimiter="\t"):
                idx = int(row["index"])
                records[idx].update(
                    status="benchmarked" if row["status"] == "ok" else "benchmark_" + row["status"],
                    latency_ms=float(row["latency_ms"]) if row["latency_ms"] else None,
                    error=row["error"] or None,
                )
    for rec in records:
        if rec["status"] not in TERMINAL:
            rec.update(status="worker_failed", error="no terminal benchmark outcome")
    write_json(output / "outcomes.json", records)


def main(argv=None, *, family=None):
    if argv is None and len(sys.argv) == 4 and sys.argv[1] == "--worker":
        output = Path(sys.argv[3])
        worker(sys.argv[2], output)
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    if family is None:
        parser.add_argument("--family", choices=OPS, required=True)
    else:
        parser.set_defaults(family=family)
    parser.add_argument("--workloads", nargs="+")
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    parser.add_argument("--config-indices", type=int, nargs="+")
    parser.add_argument("--gpus", type=int, nargs="+", help="Physical GPU indices; default: all idle visible GPUs of the same model")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-spill-bytes", type=int, default=0, help="Post-compile ptxas spill limit in bytes; 0 rejects any spill")
    parser.add_argument("--max-local-bytes", type=int, default=None, help="Post-compile ptxas local-memory limit in bytes; omit to leave local traffic unlimited")
    parser.add_argument("--tiletune", action="store_true")
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--metric", choices=["memory", "traffic_waves", "pipeline_time"], default="memory")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)
    if min(args.workers, args.warmup, args.rep, args.timeout) < 1 or args.group_size < 2 or args.seed < 0:
        parser.error("positive budgets, group-size >= 2 and nonnegative seed required")
    if args.max_spill_bytes < 0 or (args.max_local_bytes is not None and args.max_local_bytes < 0):
        parser.error("spill/local byte limits must be nonnegative")
    if args.alpha is not None:
        if not args.tiletune:
            parser.error("--alpha requires --tiletune")
        if not math.isfinite(args.alpha) or not 0 < args.alpha <= 1:
            parser.error("--alpha must be finite and in (0, 1]")
    if args.tiletune and args.alpha is None:
        parser.error("--tiletune requires --alpha")
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    plan = system_plan(args.family, workloads=args.workloads, variants=variants, indices=args.config_indices)
    if args.plan:
        print(json.dumps(plan, indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required for execution")
    from experiments.utils.monitor import snapshot, idle_gpus, visible_gpus, run_monitored

    observed = snapshot()
    gpus = idle_gpus(observed) if args.gpus is None else [g for g in visible_gpus(observed) if int(g["index"]) in args.gpus]
    if args.gpus is not None and len(gpus) != len(set(args.gpus)):
        parser.error("requested GPUs must be visible physical indices")
    if args.gpus is None and gpus:
        gpus = [g for g in gpus if g["name"] == gpus[0]["name"]]
    if not gpus or len({g["name"] for g in gpus}) != 1:
        parser.error("select available GPUs of one model")
    if any(VARIANTS[v][2] for v in variants) and len(gpus) < 2:
        parser.error("multi_gpu and combined require at least two GPUs")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    settings = {k: getattr(args, k) for k in ("workers", "warmup", "rep", "timeout", "group_size", "seed", "tiletune", "alpha", "metric", "max_spill_bytes", "max_local_bytes")}
    write_json(root / "plan.json", dict(cases=plan, settings=settings, gpus=gpus))
    rows = []
    for item in plan:
        output = root / item["workload"]["name"] / item["variant"]
        output.mkdir(parents=True)
        request = output / "request.json"
        write_json(request, dict(**item, settings=settings))
        active = gpus if VARIANTS[item["variant"]][2] else gpus[:1]
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=",".join(g["uuid"] for g in active),
            TILELANG_DISABLE_CACHE="1",
            TILELANG_AUTO_TUNING_DISABLE_CACHE="1",
            TILELANG_AUTO_TUNING_CPU_COUNTS=str(args.workers),
            TILELANG_AUTOTUNE_TIMING_LOG=str(output / "timings.tsv"),
        )
        try:
            run_monitored(
                [sys.executable, "-m", "experiments.common.system", "--worker", str(request), str(output)],
                output,
                active,
                env=env,
                cwd=Path(__file__).resolve().parents[2],
            )
        except BaseException:
            if (output / "summary.json").exists():
                (output / "summary.json").rename(output / "discarded-summary.json")
            raise
        row = json.loads((output / "summary.json").read_text())
        baseline = next((r for r in rows if r["workload"] == row["workload"] and r["variant"] == "baseline"), None)
        if baseline and "tuning_seconds" in baseline and "tuning_seconds" in row:
            row["tuning_speedup_vs_baseline"] = baseline["tuning_seconds"] / row["tuning_seconds"]
        else:
            row["tuning_speedup_vs_baseline"] = 1.0 if row["variant"] == "baseline" and "tuning_seconds" in row else None
        rows.append(row)
        write_json(root / "comparison.json", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
