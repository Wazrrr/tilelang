"""Collect disjoint shape splits, freeze models, and compare a portable matrix.

python -m experiments.portable.compare --device ampere --plan
python -m experiments.portable.compare --device ampere --output experiments/results/a100 --wait-idle

Every method uses one kernel, candidate grid, seed, and timing backend per case.
Selection runs before test oracles. Model fitting never reads test measurements.
"""

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

from .run import make_request, run_case, write_json
from .spec import Device, TARGETS, configurations, default_workloads, load_manifest, support_reason


def split_workloads(workloads, scales):
    from experiments.xgboost.data import canonical_workload, digest

    result, seen = {}, {}
    for split, factors in scales.items():
        result[split] = []
        for w in workloads:
            for factor in factors:
                p = dict(w.parameters)
                keys = ("m", "n", "k") if w.op == "gemm" else ("rows", "columns") if "rows" in p else ("sequence",)
                for key in keys:
                    multiple = p.get("chunk_size", 32) if key == "sequence" else 32
                    p[key] = max(multiple, int(p[key] * factor / multiple) * multiple)
                item = replace(w, name=f"{w.name}.{split}{factor:g}", parameters=p)
                key = digest(canonical_workload(item))
                if key in seen:
                    raise ValueError(f"shape split overlap: {item.name} and {seen[key]}")
                seen[key] = item.name
                result[split].append(item)
    return result


def gpu_snapshot():
    if not __import__("shutil").which("nvidia-smi"):
        return dict(available=False, reason="nvidia-smi unavailable; inspect the accelerator scheduler separately")
    ordinal = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

    def query(arguments):
        return subprocess.check_output(["nvidia-smi", "-i", ordinal, *arguments, "--format=csv,noheader,nounits"], text=True).strip()

    return dict(
        timestamp=datetime.now(timezone.utc).isoformat(),
        gpu=query(["--query-gpu=uuid,name,utilization.gpu,memory.used,clocks.sm,temperature.gpu,power.draw"]),
        processes=query(["--query-compute-apps=pid,process_name,used_memory"]),
    )


def wait_for_idle(output, *, wait=False, allow_contended=False):
    while True:
        snapshot = gpu_snapshot()
        with (output / "gpu_observations.jsonl").open("a") as stream:
            stream.write(json.dumps(snapshot) + "\n")
        # The coordinator can own a context after profiling; all other processes
        # must finish before starting the next timing session.
        processes = [line for line in snapshot.get("processes", "").splitlines() if line.split(",")[0].strip() != str(os.getpid())]
        if not processes or allow_contended:
            return
        if not wait:
            raise RuntimeError("GPU has other compute processes; use --wait-idle or explicitly --allow-contended")
        print("Waiting for an idle GPU; other compute processes: " + "; ".join(processes), flush=True)
        time.sleep(30)


def remeasure(request, methods, output, repeats):
    import torch
    import tilelang
    from .kernels import make_case
    from .spec import Workload

    workload = Workload(**request["workload"])
    case, settings = make_case(workload), request["settings"]
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = case.inputs("cuda", torch.Generator(device="cuda").manual_seed(settings["seed"]))
    expected = case.reference(*inputs)
    expected = expected if isinstance(expected, (list, tuple)) else [expected]
    kernels, samples = {}, {}
    for result in methods.values():
        if result.get("status") != "completed":
            continue
        config = result["winner"]["config"]
        key = json.dumps(config, sort_keys=True)
        if key in kernels:
            continue
        kwargs = {k: v for k, v in config.items() if k != "pass_configs"}
        kernel = tilelang.compile(
            case.build(**kwargs),
            target=request["device"]["target"],
            out_idx=case.out_idx,
            execution_backend="tvm_ffi",
            pass_configs={**case.pass_configs, **config.get("pass_configs", {})},
        )
        actual = kernel(*inputs)
        case.check(actual if isinstance(actual, (list, tuple)) else [actual], expected)
        kernels[key], samples[key] = kernel.get_profiler(), []
    rng = random.Random(settings["seed"])
    for _ in range(repeats):
        order = list(kernels)
        rng.shuffle(order)
        for key in order:
            samples[key].append(
                kernels[key].do_bench(
                    input_tensors=inputs,
                    backend="event",
                    warmup=settings["warmup"],
                    rep=settings["rep"],
                )
            )
    report = {}
    for name, result in methods.items():
        if result.get("status") == "completed":
            values = samples[json.dumps(result["winner"]["config"], sort_keys=True)]
            report[name] = dict(
                median_ms=statistics.median(values),
                samples_ms=values,
                relative_spread=(max(values) - min(values)) / statistics.median(values),
            )
    if "brute_force" in report:
        for value in report.values():
            value["performance_vs_brute_force"] = report["brute_force"]["median_ms"] / value["median_ms"]
    write_json(output, report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=TARGETS, default="ampere")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--workloads", nargs="+")
    parser.add_argument("--train-scales", nargs="+", type=float, default=[0.25, 0.5])
    parser.add_argument("--validation-scales", nargs="+", type=float, default=[0.75])
    parser.add_argument("--test-scales", nargs="+", type=float, default=[1, 2])
    parser.add_argument("--top-k", type=int, default=20, help="Maximum online budget, capped at ceil(grid_size * budget_fraction)")
    parser.add_argument("--budget-fraction", type=float, default=0.1)
    parser.add_argument("--config-indices", nargs="+", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--case-timeout", type=int, default=1800)
    parser.add_argument("--validation-repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/portable-comparison"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--wait-idle", action="store_true")
    parser.add_argument("--allow-contended", action="store_true")
    args = parser.parse_args(argv)
    if (
        not 0 < args.budget_fraction <= 1
        or any(getattr(args, k) <= 0 for k in ("top_k", "workers", "warmup", "rep", "timeout", "case_timeout", "validation_repeats"))
        or any(not math.isfinite(s) or s <= 0 for s in args.train_scales + args.validation_scales + args.test_scales)
    ):
        parser.error("budgets and scales must be positive; budget fraction must be in (0, 1]")
    if args.manifest:
        devices, workloads = load_manifest(json.loads(args.manifest.read_text()))
        devices = [
            replace(d, profiles={k: str((args.manifest.resolve().parent / v).resolve()) for k, v in d.profiles.items()})
            if d.profiles
            else d
            for d in devices
        ]
    else:
        devices, workloads = [Device(args.device, TARGETS[args.device])], default_workloads()
    if args.workloads:
        if set(args.workloads) - {w.name for w in workloads}:
            parser.error("unknown workload name")
        workloads = [w for w in workloads if w.name in args.workloads]
    splits = split_workloads(workloads, dict(train=args.train_scales, validation=args.validation_scales, test=args.test_scales))
    # Retain target-specific overrides when split names are derived from the base.
    devices = [
        replace(
            d,
            configs={
                item.name: d.configs[w.name]
                for values in splits.values()
                for item in values
                for w in workloads
                if item.name.startswith(w.name + ".") and w.name in d.configs
            },
        )
        if d.configs
        else d
        for d in devices
    ]
    settings = {k: getattr(args, k) for k in ("top_k", "config_indices", "seed", "workers", "warmup", "rep", "timeout", "case_timeout")}
    settings.update(memory_regime="streaming", trace=False)
    plan = dict(
        version=1,
        devices=[d.to_dict() for d in devices],
        splits={k: [w.to_dict() for w in v] for k, v in splits.items()},
        settings=settings,
        budget_fraction=args.budget_fraction,
        validation_repeats=args.validation_repeats,
        allow_contended=args.allow_contended,
    )
    if args.plan:
        print(json.dumps(plan, indent=2))
        return 0
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    if args.resume and (root / "plan.json").exists():
        if json.loads((root / "plan.json").read_text()) != plan:
            raise ValueError("resume plan differs from the frozen original plan")
    else:
        write_json(root / "plan.json", plan)
    print(f"Results directory: {root}", flush=True)
    os.environ.update(TILELANG_DISABLE_CACHE="1", TILELANG_AUTO_TUNING_DISABLE_CACHE="1")
    wait_for_idle(root, wait=args.wait_idle, allow_contended=args.allow_contended)
    from experiments._common import source_hashes
    from tilelang.cache.kernel_cache import KernelCache
    from tilelang.contrib.cc import get_cplus_compiler

    compiler = get_cplus_compiler()
    provenance = dict(
        source_sha256=source_hashes("experiments/portable/kernels.py"),
        native_build=KernelCache._get_tilelang_lib_stamp(),
        python=sys.version,
        host_compiler=compiler,
        host_compiler_version=subprocess.check_output([compiler, "--version"], text=True),
        cuda_home=os.environ.get("CUDA_HOME"),
    )
    if (root / "provenance.json").exists() and json.loads((root / "provenance.json").read_text()) != provenance:
        raise ValueError("source/native build changed since collection; use a new output directory")
    write_json(root / "provenance.json", provenance)

    def run(workload, device, method, output, metric="pipeline_time", model=None):
        count = len(settings["config_indices"]) if settings["config_indices"] else len(configurations(workload, device))
        budget = min(args.top_k, max(1, math.ceil(count * args.budget_fraction)))
        options = dict(settings, method=method, metric=metric, top_k=budget)
        if model:
            options["xgb_model"] = str(model)
        request = make_request(workload, device, options)
        if (output / "result.json").exists():
            if json.loads((output / "request.json").read_text()) != request:
                raise ValueError(f"resume request mismatch: {output}")
            return json.loads((output / "result.json").read_text())
        if output.exists():
            # Keep interrupted logs for inspection; never mix partial outcomes
            # into a restarted candidate table.
            output.rename(output.with_name(output.name + ".interrupted." + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")))
        wait_for_idle(root, wait=args.wait_idle, allow_contended=args.allow_contended)
        result = run_case(request, output)
        print(f"{output.relative_to(root)}: {result['status']} {result.get('reason') or ''}", flush=True)
        return result

    all_comparisons = []
    for device in devices:
        base = root / device.name
        base.mkdir(exist_ok=True)
        if device.target["kind"] == "cuda" and not device.profiles and not device.performance_model:
            from tilelang.tiletune import current_target, profile_device
            from .run import targets_match

            if not targets_match(device.target, current_target()):
                raise ValueError("requested device does not match this server")
            path = base / "primitive-profile.json"
            started = time.perf_counter()
            profiles = {}
            for dtype in sorted({w.dtype for w in workloads if not support_reason(w, device)}):
                profile_device(input_dtype=dtype, cache_path=path, memory_regime="streaming")
                profiles[dtype] = str(path)
            device = replace(device, profiles=profiles)
            preparation = base / "profile-preparation.json"
            if not preparation.exists():
                write_json(
                    preparation,
                    dict(
                        seconds=time.perf_counter() - started,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        note="Fixed primitives only; no workload latency anchors",
                    ),
                )
            import torch

            torch.cuda.empty_cache()
        collected = defaultdict(lambda: defaultdict(list))
        for split in ("train", "validation"):
            for w in splits[split]:
                output = base / split / w.name
                result = run(w, device, "brute_force", output)
                if result["status"] == "completed":
                    collected[w.op][split].append(output)
        models = {}
        from experiments.xgboost.data import read_runs
        from experiments.xgboost.model import train

        for op, paths in collected.items():
            model = base / "models" / (op + ".json")
            if not paths["train"] or not paths["validation"]:
                continue
            try:
                if not model.exists():
                    train(read_runs(paths["train"]), read_runs(paths["validation"]), model, seed=args.seed, workers=args.workers)
                models[op] = model
            except Exception as error:
                write_json(base / f"xgboost-{op}-failure.json", dict(reason=f"{type(error).__name__}: {error}"))
        for w in splits["test"]:
            output = base / "test" / w.name
            output.mkdir(parents=True, exist_ok=args.resume)
            methods, reports = {}, {}
            # Both analytical metrics are declared in advance. Neither is chosen
            # after reading this workload's oracle or validation measurements.
            for label, method, metric in (
                ("tiletune", "top_k", "pipeline_time"),
                ("tiletune_traffic", "top_k", "traffic_waves"),
                ("carver", "carver", "pipeline_time"),
                ("xgboost", "xgboost", "pipeline_time"),
                ("brute_force", "brute_force", "pipeline_time"),
            ):
                if method == "brute_force" and not (output / "frozen-rankings.json").exists():
                    frozen = {name: dict(ranking=report["ranking"], selection=report.get("selection")) for name, report in reports.items()}
                    write_json(
                        output / "frozen-rankings.json",
                        dict(
                            recorded_at=datetime.now(timezone.utc).isoformat(),
                            sha256=hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest(),
                            methods=frozen,
                        ),
                    )
                if method == "xgboost" and w.op not in models:
                    methods[label] = dict(status="unavailable", reason="no completed training/validation data for this operation")
                    continue
                methods[label] = run(w, device, method, output / label, metric, models.get(w.op) if method == "xgboost" else None)
                report_path = output / label / ("tiletune.json" if method == "top_k" else method + ".json")
                if report_path.exists():
                    reports[label] = json.loads(report_path.read_text())
            oracle_path = output / "brute_force" / "outcomes.json"
            oracle = json.loads(oracle_path.read_text()) if oracle_path.exists() else []
            from .diagnostics import assess

            diagnostics = {label: assess(report, oracle, seed=args.seed) for label, report in reports.items() if label != "brute_force"}
            write_json(output / "methods.json", methods)
            request_path = output / "brute_force" / "request.json"
            validation = output / "validation.json"
            if device.worker and not validation.exists():
                write_json(validation, dict(status="unsupported", reason="external worker must supply its own winner remeasurement"))
            if request_path.exists() and not validation.exists() and any(r.get("status") == "completed" for r in methods.values()):
                wait_for_idle(root, wait=args.wait_idle, allow_contended=args.allow_contended)
                command = [
                    sys.executable,
                    "-m",
                    "experiments.portable.compare",
                    "--remeasure",
                    str(request_path),
                    str(output / "methods.json"),
                    str(validation),
                    str(args.validation_repeats),
                ]
                try:
                    with (output / "validation.log").open("w") as log:
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.case_timeout, check=True)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                    write_json(validation, dict(status="failed", reason=str(error)))
            comparison = dict(
                workload=w.to_dict(),
                device=device.name,
                methods=methods,
                diagnostics=diagnostics,
                validation=json.loads(validation.read_text()) if validation.exists() else None,
            )
            write_json(output / "comparison.json", comparison)
            all_comparisons.append(comparison)
            write_json(root / "comparison.json", dict(version=1, results=all_comparisons))
    return int(any(row["status"] == "failed" for comparison in all_comparisons for row in comparison["methods"].values()))


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "--remeasure":
        remeasure(json.loads(Path(sys.argv[2]).read_text()), json.loads(Path(sys.argv[3]).read_text()), Path(sys.argv[4]), int(sys.argv[5]))
    else:
        raise SystemExit(main())
