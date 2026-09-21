"""System ablations on the same example kernels and pools as TileTune."""

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from threading import Lock

from experiments.utils.io import write_json
from experiments.common.spec import Workload
from experiments.families import FAMILIES, family_module

VARIANTS = {
    "baseline": (False, False, False),
    "pipeline": (True, False, False),
    "grouped": (False, True, False),
    "multi_gpu": (False, False, True),
    "combined": (True, True, True),
}
OPS = {family: [op for op in FAMILIES if FAMILIES[op] == family] for family in dict.fromkeys(FAMILIES.values())}


def variant_options(variant):
    return (True, True, True) if variant == "tiletune" else VARIANTS[variant]


def system_plan(family, *, workloads=None, variants=None, indices=None):
    cases = [w for op in OPS[family] for w in family_module(op, "cases").cases(holdout=True)]
    if workloads:
        if set(workloads) - {w.name for w in cases}:
            raise ValueError("unknown final workload name for this family")
        cases = [w for w in cases if w.name in workloads]
    from experiments.utils.cli import select_configs

    return [
        dict(
            workload=w.to_dict(),
            variant=v,
            indices=select_configs(family_module(w.op, "spaces").get_configs(), indices)[0],
        )
        for w in cases
        for v in (variants or VARIANTS)
    ]


def worker(request_path, output):
    from experiments.utils.imports import use_local_tilelang

    use_local_tilelang()
    request = json.loads(Path(request_path).read_text())
    if request.get("cpu_ids"):
        from experiments.utils.isolation import prepare_worker

        prepare_worker(request["cpu_ids"], request["parent_pid"])
    if request.get("source_identity"):
        from .h200 import source_identity

        if source_identity() != request["source_identity"]:
            raise RuntimeError("code or native build changed after the study was frozen")
    import torch
    from tilelang.autotuner import AutoTuner, set_autotune_inputs
    from tilelang.tiletune import current_target
    from experiments.utils.cli import device_info, source_hashes, select_configs, observe_compilation
    from .kernels import make_case

    w, settings = Workload(**request["workload"]), request["settings"]
    pipeline, grouped, multi_gpu = variant_options(request["variant"])
    benchmark_backend = settings.get("benchmark_backend", "event")
    if benchmark_backend not in ("event", "cupti", "cudagraph"):
        raise ValueError("unknown benchmark backend")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    devices = list(range(torch.cuda.device_count())) if multi_gpu else [0]
    if len(devices) > 4 or request.get("gpu_count", len(devices)) != len(devices):
        raise ValueError("worker requires the frozen GPU count, at most four")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    case = make_case(w)
    pool = family_module(w.op, "spaces").get_configs()
    indices, configs = select_configs(pool, request["indices"])
    analytical = request["variant"] == "tiletune"
    if analytical and indices != list(range(len(pool))):
        raise ValueError("TileTune alpha requires the complete original pool")
    if request.get("configs", configs) != configs:
        raise ValueError("configuration pool changed after freezing the request")
    inputs = case.inputs("cuda", torch.Generator(device="cuda").manual_seed(settings["seed"]))
    case.check_input_values(inputs)
    references = {d: case.reference(*(value.to(f"cuda:{d}") for value in inputs)) for d in devices}
    for d in devices:
        torch.cuda.synchronize(d)
    outcomes = {}
    journal_lock = Lock()

    class ObservedTuner(AutoTuner):
        def _resolve_num_compile_workers(self):
            count = super()._resolve_num_compile_workers()
            if count != settings["workers"]:
                raise RuntimeError(f"requested {settings['workers']} compiler workers, resolved {count}")
            return count

        def _prepare_compile_execution(self, *args, **kwargs):
            if self.tiletune_session is not None:
                # Selection is final before any compilation; preserve it on interruption.
                write_json(output / "selection.json", self.tiletune_session.selection)
                self.tiletune_session.finish()
            if settings.get("preflight"):
                chosen = kwargs["config_indices"][:8]
                write_json(output / "preflight-indices.json", chosen)
                if self.tiletune_session is not None:
                    for index in kwargs["config_indices"][8:]:
                        self.tiletune_session.records[index]["status"] = "preflight_omitted"
                kwargs["config_indices"] = chosen
            execution = observe_compilation(super()._prepare_compile_execution(*args, **kwargs), outcomes)
            for future, items in execution[2].items():

                def checkpoint(done, items=items):
                    with journal_lock, (output / "compilation-events.jsonl").open("a") as stream:
                        for index, _ in items:
                            stream.write(json.dumps(dict(index=index, **outcomes[index])) + "\n")

                future.add_done_callback(checkpoint)
            return execution

    target = current_target()
    from tilelang.contrib.cc import get_cplus_compiler
    from tilelang.contrib.nvcc import find_cuda_path

    environment = dict(
        python=sys.version,
        torch=torch.__version__,
        cuda_runtime=torch.version.cuda,
        host_compiler=subprocess.check_output([get_cplus_compiler(), "--version"], text=True).strip(),
        device_compiler=subprocess.check_output([str(Path(find_cuda_path()) / "bin/nvcc"), "--version"], text=True).strip(),
        driver=subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"], text=True).strip(),
    )
    with set_autotune_inputs(inputs):
        tuner = (
            ObservedTuner(case.build, configs)
            .set_compile_args(out_idx=case.out_idx, pass_configs=case.pass_configs, target=target, execution_backend="tvm_ffi")
            .set_profile_args(
                ref_prog=lambda *values: references[values[0].device.index],
                manual_check_prog=case.check,
                backend=benchmark_backend,
                cache_input_tensors=False,
                rtol=case.rtol,
                atol=case.atol,
                max_mismatched_ratio=0.0,
            )
            .set_benchmark_report_path(str(output / "benchmarks.tsv"))
        )
    if analytical:
        from tilelang.tiletune import TileTuneConfig, query_device_limits
        from .resource_policy import h200_post_compile_policy

        tuner.set_tiletune_args(
            TileTuneConfig(
                enabled=True,
                mode="report_only",
                ranking_metric="memory",
                alpha=0.5,
                memory_diagnostics=False,
                input_values=case.input_values or None,
                device_limits=query_device_limits(target),
                max_spill_bytes=None,
                max_local_bytes=None,
                post_compile_policy=h200_post_compile_policy(w, target),
                report_path=str(output / "tiletune.json"),
            )
        )
    write_json(
        output / "experiment.json",
        dict(
            request,
            configs=configs,
            original_indices=indices,
            target=target,
            source_sha256=source_hashes("experiments/common/system.py"),
            devices=device_info(devices),
            environment=environment,
            measurement=dict(
                backend=benchmark_backend,
                warmup_ms=settings["warmup"],
                rep_ms=settings["rep"],
                cache_flush_bytes=256 * 1024 * 1024,
            ),
            cold_kernel_cache=True,
            cold_autotune_cache=True,
        ),
    )
    started = time.perf_counter()
    try:
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
    finally:
        write_json(output / "compilation.json", outcomes)
    duration = time.perf_counter() - started
    if request.get("source_identity") and source_identity() != request["source_identity"]:
        raise RuntimeError("code or native build changed during the workload; discard this attempt")
    from .spaces import config_id

    records = (
        tuner.tiletune_report["configs"]
        if analytical
        else [dict(index=i, config=config, **outcomes.get(i, dict(status="not_attempted"))) for i, config in enumerate(configs)]
    )
    with (output / "benchmarks.tsv").open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            record = records[int(row["index"])]
            record.update(
                status="benchmarked" if row["status"] == "ok" else "benchmark_" + row["status"],
                latency_ms=float(row["latency_ms"]) if row["latency_ms"] else None,
                error=row["error"] or None,
            )
    from experiments.utils.results import TERMINAL

    for record in records:
        record.update(original_index=indices[record["index"]], config_id=config_id(record["config"]))
        allowed = TERMINAL | {"not_selected"} | ({"preflight_omitted"} if settings.get("preflight") else set())
        if record["status"] not in allowed:
            raise RuntimeError(f"incomplete workload: candidate {record['index']} is {record['status']}")
    write_json(output / "outcomes.json", records)
    write_json(output / "compilation.json", outcomes)
    write_json(
        output / "summary.json",
        dict(
            status="completed",
            workload=w.name,
            variant=request["variant"],
            tuning_seconds=duration,
            compiler_workers=settings["workers"],
            benchmark_gpu_count=len(devices),
            candidate_statuses=dict(Counter(record["status"] for record in records)),
            selection=tuner.tiletune_report["selection"] if analytical else None,
            config_count=len(configs),
            winner_config=result.config,
            winner_latency_ms=result.latency,
        ),
    )


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
    parser.add_argument("--gpus", type=int, nargs="+", help="At most four physical GPU indices; default: up to four idle matching GPUs")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)
    if args.gpus and (len(args.gpus) > 4 or len(set(args.gpus)) != len(args.gpus)):
        parser.error("select at most four distinct GPUs")
    if min(args.workers, args.warmup, args.rep, args.timeout) < 1 or args.group_size < 2 or args.seed < 0:
        parser.error("positive budgets, group-size >= 2 and nonnegative seed required")
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
        gpus = [g for g in gpus if g["name"] == gpus[0]["name"]][:4]
    if not gpus or len({g["name"] for g in gpus}) != 1:
        parser.error("select available GPUs of one model")
    if any(VARIANTS[v][2] for v in variants) and len(gpus) < 2:
        parser.error("multi_gpu and combined require at least two GPUs")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    settings = {k: getattr(args, k) for k in ("workers", "warmup", "rep", "timeout", "group_size", "seed")}
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
            TILELANG_AUTO_TUNING_MAX_CPU_COUNT=str(args.workers),
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
        row["tuning_speedup_vs_baseline"] = (
            baseline["tuning_seconds"] / row["tuning_seconds"] if baseline else 1.0 if row["variant"] == "baseline" else None
        )
        rows.append(row)
        write_json(root / "comparison.json", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
