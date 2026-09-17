"""Collect disjoint shape splits, freeze models, and compare a portable matrix.

python -m experiments.common.comparison --device ampere --plan
python -m experiments.common.comparison --device ampere --output experiments/results/a100 --wait-idle

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

from .run import make_request, run_case

from experiments.utils.io import write_json
from .spec import PRESETS
from .spec import Device, TARGETS, Workload, configurations, default_workloads, load_manifest, support_reason
from experiments.xgboost.data import canonical_workload
from experiments.xgboost.model import DEFAULT_LEARNING_RATE, DEFAULT_MAX_DEPTH, DEFAULT_ROUNDS, DEFAULT_SUBSAMPLE, EARLY_STOPPING_ROUNDS
from experiments.xgboost.sampling import DEFAULT_SAMPLE_FRACTION, SAMPLING_POLICY, STRATIFIED_POLICY, sampling_plan


def training_sample(workload, device, *, fraction, seed, config_indices=None, policy=SAMPLING_POLICY):
    """Select collection indices before any outcomes or timings are available."""
    configs = configurations(workload, device)
    if config_indices is None:
        indices = list(range(len(configs)))
    else:
        from experiments.utils.cli import select_configs

        indices, configs = select_configs(configs, config_indices)
    plan = sampling_plan(canonical_workload(workload), configs, fraction=fraction, seed=seed, policy=policy)
    return dict(config_indices=[indices[i] for i in plan["selected_indices"]], xgb_sampling=plan)


def split_workloads(workloads, scales):
    from experiments.xgboost.data import canonical_workload, digest

    result, seen = {}, {}
    for split, factors in scales.items():
        result[split] = []
        for w in workloads:
            for factor in factors:
                p = dict(w.parameters)
                keys = ("m", "n", "k") if w.op in ("gemm", "gemm_fp8") else ("rows", "columns") if "rows" in p else ("sequence",)
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
    expected = expected if isinstance(expected, list | tuple) else [expected]
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
        case.check(actual if isinstance(actual, list | tuple) else [actual], expected)
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
    parser.add_argument("--config-space", choices=PRESETS)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("tiletune", "tiletune_exploration", "tiletune_traffic", "carver", "xgboost", "xgboost_stratified", "random"),
        default=["tiletune", "tiletune_traffic", "carver", "xgboost"],
        help="Methods declared before collection; the independent brute-force oracle always runs last",
    )
    parser.add_argument("--exploration-fraction", type=float, default=0.2)
    parser.add_argument("--split-manifest", type=Path, help="Explicit, disjoint train/validation/test workloads; retains tail dimensions")
    parser.add_argument("--train-scales", nargs="+", type=float, help="Explicit scaled-shape study; default uses family splits")
    parser.add_argument("--validation-scales", nargs="+", type=float)
    parser.add_argument("--test-scales", nargs="+", type=float)
    parser.add_argument("--top-k", type=int, default=20, help="Maximum online budget, capped at ceil(grid_size * budget_fraction)")
    parser.add_argument("--budget-fraction", type=float, default=0.1)
    parser.add_argument(
        "--xgb-sample-fraction",
        type=float,
        default=DEFAULT_SAMPLE_FRACTION,
        help="Fraction of each training/validation pool to collect for XGBoost (default: 0.1)",
    )
    parser.add_argument("--config-indices", nargs="+", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--case-timeout", type=int, default=1800)
    parser.add_argument("--validation-repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--input-seed", type=int, default=123)
    parser.add_argument("--phase", choices=("all", "selection", "oracle"), default="all")
    parser.add_argument("--oracle-root", type=Path, help="Shared oracle directory, used only after all seed selections are frozen")
    parser.add_argument("--measurement-identity", type=Path, help="Verified measurement identity supplied by the baseline coordinator")
    parser.add_argument("--skip-profile", action="store_true", help="Baseline-only collection needs no TileTune primitive profile")
    parser.add_argument("--skip-validation", action="store_true", help="Use the oracle table without additional winner reruns")
    parser.add_argument("--output", type=Path, default=Path("experiments/results/portable-comparison"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--wait-idle", action="store_true")
    parser.add_argument("--allow-contended", action="store_true")
    args = parser.parse_args(argv)
    if (
        not 0 < args.exploration_fraction <= 1
        or not 0 < args.budget_fraction <= 1
        or not 0 < args.xgb_sample_fraction <= 1
        or args.seed < 0
        or any(getattr(args, k) <= 0 for k in ("top_k", "workers", "warmup", "rep", "timeout", "case_timeout", "validation_repeats"))
        or any(
            not math.isfinite(s) or s <= 0 for values in (args.train_scales, args.validation_scales, args.test_scales) for s in values or []
        )
    ):
        parser.error("budgets and scales must be positive; fractions must be in (0, 1]; seed must be nonnegative")
    if args.manifest and args.split_manifest:
        parser.error("choose --manifest or --split-manifest")
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
    if args.config_space:
        workloads = [replace(w, config_space=args.config_space) for w in workloads]
    if args.split_manifest:
        bundle = json.loads(args.split_manifest.read_text())
        if set(bundle) != {"version", "devices", "splits"} or set(bundle["splits"]) != {"train", "validation", "test"}:
            raise ValueError("split manifest requires explicit train/validation/test workloads")
        splits = {}
        for name, items in bundle["splits"].items():
            # Validate override names against all splits, not each split alone.
            all_items = [w for values in bundle["splits"].values() for w in values]
            devices, _ = load_manifest(dict(version=bundle["version"], devices=bundle["devices"], workloads=all_items))
            splits[name] = [Workload(**item) for item in items]
        from experiments.xgboost.data import digest

        keys = [digest(canonical_workload(w)) for values in splits.values() for w in values]
        if len(keys) != len(set(keys)):
            raise ValueError("shape split overlap in explicit manifest")
        workloads = [w for values in splits.values() for w in values]
        devices = [
            replace(d, profiles={k: str((args.split_manifest.resolve().parent / v).resolve()) for k, v in d.profiles.items()})
            if d.profiles
            else d
            for d in devices
        ]
    elif args.manifest or any(values is not None for values in (args.train_scales, args.validation_scales, args.test_scales)):
        splits = split_workloads(
            workloads,
            dict(train=args.train_scales or [0.25, 0.5], validation=args.validation_scales or [0.75], test=args.test_scales or [1, 2]),
        )
    else:
        from experiments.families import family_module

        splits = dict(train=[], validation=[], test=workloads)
        for op in dict.fromkeys(w.op for w in workloads):
            a, b, validation = family_module(op, "cases").training_cases()
            splits["train"].extend((a, b))
            splits["validation"].append(validation)
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
        if d.configs and not args.split_manifest
        else d
        for d in devices
    ]
    settings = {k: getattr(args, k) for k in ("top_k", "config_indices", "seed", "workers", "warmup", "rep", "timeout", "case_timeout")}
    settings.update(memory_regime="streaming", trace=False, seed=args.input_seed)
    if args.measurement_identity:
        settings["measurement_identity"] = json.loads(args.measurement_identity.read_text())
    if args.skip_profile and any(m.startswith("tiletune") for m in args.methods):
        parser.error("TileTune requires profile preparation")
    xgb_training = dict(
        rounds=DEFAULT_ROUNDS, max_depth=DEFAULT_MAX_DEPTH, learning_rate=DEFAULT_LEARNING_RATE, subsample=DEFAULT_SUBSAMPLE
    )
    plan = dict(
        version=1,
        devices=[d.to_dict() for d in devices],
        splits={k: [w.to_dict() for w in v] for k, v in splits.items()},
        settings=settings,
        budget_fraction=args.budget_fraction,
        oracle_root=str(args.oracle_root.resolve()) if args.oracle_root else None,
        xgb_sampling=dict(policy=SAMPLING_POLICY, fraction=args.xgb_sample_fraction, seed=args.seed),
        xgb_training=dict(**xgb_training, early_stopping_rounds=EARLY_STOPPING_ROUNDS),
        validation_repeats=args.validation_repeats,
        allow_contended=args.allow_contended,
        methods=args.methods,
        exploration_fraction=args.exploration_fraction if "tiletune_exploration" in args.methods else 0,
        xgb_sampling_policies=[
            p for label, p in (("xgboost", SAMPLING_POLICY), ("xgboost_stratified", STRATIFIED_POLICY)) if label in args.methods
        ],
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
    from experiments.utils.cli import source_hashes
    from tilelang.cache.kernel_cache import KernelCache
    from tilelang.contrib.cc import get_cplus_compiler

    compiler = get_cplus_compiler()
    provenance = dict(
        source_sha256=source_hashes("experiments/common/kernels.py"),
        native_build=KernelCache._get_tilelang_lib_stamp(),
        python=sys.version,
        host_compiler=compiler,
        host_compiler_version=subprocess.check_output([compiler, "--version"], text=True),
        cuda_home=os.environ.get("CUDA_HOME"),
    )
    if (root / "provenance.json").exists() and json.loads((root / "provenance.json").read_text()) != provenance:
        raise ValueError("source/native build changed since collection; use a new output directory")
    write_json(root / "provenance.json", provenance)

    def run(workload, device, method, output, metric="pipeline_time", model=None, sampling=None, exploration=0):
        indices = settings["config_indices"]
        if indices is None and device.subsets:
            indices = device.subsets.get(workload.name)
        count = len(indices) if indices is not None else len(configurations(workload, device))
        budget = min(args.top_k, max(1, math.ceil(count * args.budget_fraction)))
        options = dict(settings, method=method, metric=metric, top_k=budget, config_indices=indices)
        if exploration:
            options.update(exploration_fraction=exploration, selection_seed=args.seed)
        if method == "random":
            options["selection_seed"] = args.seed
        if sampling is not None:
            options.update(sampling)
        if model:
            options["xgb_model"] = str(model)
        request = make_request(workload, device, options)
        if (output / "result.json").exists():
            if json.loads((output / "request.json").read_text()) != request:
                raise ValueError(f"resume request mismatch: {output}")
            return json.loads((output / "result.json").read_text())
        if args.phase == "oracle" and (method != "brute_force" or sampling is not None):
            raise ValueError(f"oracle phase requires completed collection/selection: {output}")
        if output.exists():
            # Keep interrupted logs for inspection; never mix partial outcomes
            # into a restarted candidate table.
            output.rename(output.with_name(output.name + ".interrupted." + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")))
        wait_for_idle(root, wait=args.wait_idle, allow_contended=args.allow_contended)
        result = run_case(request, output)
        print(f"{output}: {result['status']} {result.get('reason') or ''}", flush=True)
        return result

    all_comparisons = []
    for device in devices:
        base = root / device.name
        base.mkdir(exist_ok=True)
        if not args.skip_profile and device.target["kind"] == "cuda" and not device.profiles and not device.performance_model:
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
        models = {}
        for label, policy in (("xgboost", SAMPLING_POLICY), ("xgboost_stratified", STRATIFIED_POLICY)):
            if label not in args.methods:
                continue
            collected = defaultdict(lambda: defaultdict(list))
            for split in ("train", "validation"):
                for w in splits[split]:
                    output = base / label / split / w.name
                    sample = training_sample(
                        w,
                        device,
                        fraction=args.xgb_sample_fraction,
                        seed=args.seed,
                        config_indices=settings["config_indices"]
                        if settings["config_indices"] is not None
                        else (device.subsets or {}).get(w.name),
                        policy=policy,
                    )
                    result = run(w, device, "brute_force", output, sampling=sample)
                    if result["status"] == "completed":
                        collected[w.op][split].append(output)
            from experiments.xgboost.data import read_runs
            from experiments.xgboost.model import train

            for op, paths in collected.items():
                model = base / "models" / (label + "-" + op + ".json")
                if not paths["train"] or not paths["validation"]:
                    continue
                try:
                    if not model.exists():
                        if args.phase == "oracle":
                            raise ValueError(f"oracle phase requires a frozen model: {model}")
                        train(
                            read_runs(paths["train"]),
                            read_runs(paths["validation"]),
                            model,
                            sample_fraction=args.xgb_sample_fraction,
                            sampling_policy=policy,
                            seed=args.seed,
                            workers=args.workers,
                            **xgb_training,
                        )
                    models[label, op] = model
                except Exception as error:
                    if args.phase == "oracle":
                        raise
                    write_json(base / f"{label}-{op}-failure.json", dict(reason=f"{type(error).__name__}: {error}"))
        for w in splits["test"]:
            output = base / "test" / w.name
            if args.phase == "oracle" and not (output / "frozen-rankings.json").exists():
                raise ValueError(f"oracle phase requires frozen selections: {output}")
            output.mkdir(parents=True, exist_ok=args.resume)
            methods, reports = {}, {}
            oracle_output = (
                (args.oracle_root.resolve() / device.name / "test" / w.name / "brute_force") if args.oracle_root else output / "brute_force"
            )
            # Both analytical metrics are declared in advance. Neither is chosen
            # after reading this workload's oracle or validation measurements.
            for label, method, metric in (
                ("tiletune", "top_k", "pipeline_time"),
                ("tiletune_exploration", "top_k", "pipeline_time"),
                ("tiletune_traffic", "top_k", "traffic_waves"),
                ("carver", "carver", "pipeline_time"),
                ("xgboost", "xgboost", "pipeline_time"),
                ("xgboost_stratified", "xgboost", "pipeline_time"),
                ("random", "random", "pipeline_time"),
                ("brute_force", "brute_force", "pipeline_time"),
            ):
                if method == "brute_force" and args.phase == "selection":
                    continue
                if method != "brute_force" and label not in args.methods:
                    continue
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
                if method == "xgboost" and (label, w.op) not in models:
                    methods[label] = dict(status="unavailable", reason="no completed training/validation data for this operation")
                    continue
                method_output = oracle_output if method == "brute_force" else output / label
                methods[label] = run(
                    w,
                    device,
                    method,
                    method_output,
                    metric,
                    models.get((label, w.op)) if method == "xgboost" else None,
                    exploration=args.exploration_fraction if label == "tiletune_exploration" else 0,
                )
                report_path = method_output / ("tiletune.json" if method == "top_k" else method + ".json")
                if report_path.exists():
                    reports[label] = json.loads(report_path.read_text())
            frozen = {
                name: dict(ranking=report["ranking"], selection=report.get("selection"))
                for name, report in reports.items()
                if name != "brute_force"
            }
            frozen_path = output / "frozen-rankings.json"
            if frozen_path.exists():
                if json.loads(frozen_path.read_text())["methods"] != frozen:
                    raise ValueError(f"selections changed after freezing: {output}")
            else:
                write_json(
                    frozen_path,
                    dict(
                        recorded_at=datetime.now(timezone.utc).isoformat(),
                        sha256=hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest(),
                        methods=frozen,
                    ),
                )
            oracle_path = oracle_output / "outcomes.json"
            oracle = json.loads(oracle_path.read_text()) if args.phase != "selection" and oracle_path.exists() else []
            from experiments.utils.diagnostics import assess

            diagnostics = {label: assess(report, oracle, seed=args.seed) for label, report in reports.items() if label != "brute_force"}
            write_json(output / "methods.json", methods)
            request_path = oracle_output / "request.json"
            validation = output / "validation.json"
            if (
                args.phase != "selection"
                and not args.skip_validation
                and request_path.exists()
                and not validation.exists()
                and any(r.get("status") == "completed" for r in methods.values())
            ):
                wait_for_idle(root, wait=args.wait_idle, allow_contended=args.allow_contended)
                request = make_request(
                    w, device, dict(settings, method="remeasure", methods=methods, validation_repeats=args.validation_repeats)
                )
                from experiments.suite import _existing_or_run

                measured = _existing_or_run(request, output / "winner-remeasurement")
                write_json(
                    validation,
                    measured["validation"]
                    if measured["status"] == "remeasured"
                    else dict(status=measured["status"], reason=measured.get("reason")),
                )
            comparison = dict(
                workload=w.to_dict(),
                device=device.name,
                methods=methods,
                oracle=dict(path=os.path.relpath(oracle_path, output), sha256=hashlib.sha256(oracle_path.read_bytes()).hexdigest())
                if oracle_path.is_file()
                else None,
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
