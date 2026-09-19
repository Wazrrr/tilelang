"""Fill missing measurements on idle GPUs, freeze baselines, and compare TileTune.

python -m experiments.cached_study --device hopper --output experiments/results/studies/current
Reusing the output resumes incomplete work; a new output also reuses family caches.
"""

import argparse
from collections import deque
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.common.run import make_request, run_case, validate_result
from experiments.common.spec import Device, TARGETS, Workload, configuration_space
from experiments.families import FAMILIES
from experiments.suite import DEVICE_PATTERNS, study_plan
from experiments.utils.baseline_store import (
    hash_files,
    identities,
    load_bundle,
    publish_bundle,
    reuse_identity,
    storage_root,
    validate_case_bundle,
    verify_bundle,
)
from experiments.utils.io import write_json
from experiments.utils.monitor import idle_gpus, matches_cuda_device, snapshot
from experiments.utils.results import load_oracle, read
from experiments.xgboost.data import digest

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = dict(
    workers=8,
    warmup=10,
    rep=50,
    timeout=30,
    case_timeout=7200,
    top_k=20,
    metric="pipeline_time",
    memory_regime="streaming",
    trace=False,
    seed=123,
)


def successful_record(path):
    """A result is reusable only after the contention monitor has finalized."""
    if not (path / "result.json").is_file() or not (path / "monitor.json").is_file():
        return False
    result, audit = read(path / "result.json"), read(path / "monitor.json")
    return result["status"] != "failed" and audit.get("status") == "uncontended"


def run_job(job):
    output = Path(job["output"])
    request = make_request(Workload(**job["workload"]), Device(**job["device"]), job["settings"])
    if successful_record(output):
        if read(output / "request.json") != request:
            raise ValueError(f"Cached request differs: {output}")
        validate_result(read(output / "result.json"), request)
        return 0
    if output.exists():
        output.rename(output.with_name(output.name + ".discarded." + str(time.time_ns())))
    result = run_case(request, output)
    if successful_record(output):
        return 0
    monitor = read(output / "monitor.json") if (output / "monitor.json").exists() else {}
    if monitor.get("status") in ("contended", "monitor_gap", "host_contended") or "no matching idle visible CUDA GPU" in result.get(
        "reason", ""
    ):
        return 75  # The coordinator puts the job back on an idle GPU.
    print(json.dumps(result), flush=True)
    return 1


def run_jobs(jobs, root, device, gpus=None):
    """One measured invocation per GPU; newly idle matching GPUs join the queue."""
    pending, active, errors = deque(), {}, []
    root.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        output = Path(job["output"])
        if successful_record(output):
            expected = make_request(Workload(**job["workload"]), Device(**job["device"]), job["settings"])
            if read(output / "request.json") != expected:
                raise ValueError(f"Cached request differs: {output}")
            validate_result(read(output / "result.json"), expected)
            continue
        path = root / (job["name"] + ".json")
        write_json(path, job)
        pending.append(path)
    try:
        while pending or active:
            for uuid, (process, path, stream) in list(active.items()):
                if process.poll() is None:
                    continue
                stream.close()
                del active[uuid]
                if process.returncode == 75:
                    pending.append(path)
                elif process.returncode:
                    errors.append(dict(job=path.stem, exit_code=process.returncode))
            for gpu in idle_gpus(snapshot()):
                if not pending:
                    break
                if gpu["uuid"] in active or not matches_cuda_device(gpu, device) or (gpus and int(gpu["index"]) not in gpus):
                    continue
                path = pending.popleft()
                stream = path.with_suffix(".log").open("a")
                env = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=gpu["uuid"],
                    CUDA_DEVICE_ORDER="PCI_BUS_ID",
                    OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                )
                process = subprocess.Popen(
                    [sys.executable, "-m", "experiments.cached_study", "--job", str(path)],
                    cwd=ROOT,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
                active[gpu["uuid"]] = process, path, stream
                print(f"START {path.stem} GPU {gpu['index']} PID {process.pid}", flush=True)
            write_json(
                root / "status.json",
                dict(
                    pending=[p.stem for p in pending],
                    active={u: dict(pid=p.pid, job=f.stem) for u, (p, f, _) in active.items()},
                    errors=errors,
                ),
            )
            if pending or active:
                time.sleep(2)
    finally:
        for process, _, stream in active.values():
            # Each worker's monitor owns its child process group and stops it on exit.
            process.terminate()
            process.wait()
            stream.close()
    if errors:
        raise RuntimeError(f"Failed jobs remain visible in {root / 'status.json'}: {errors}")


def job(name, workload, device, output, method, **settings):
    return dict(
        name=name,
        workload=workload.to_dict(),
        device=device.to_dict(),
        output=str(output),
        settings=dict(SETTINGS, method=method, **settings),
    )


def find_bundle(location, identity):
    """Search current and historical complete bundles; never refresh a valid one."""
    paths = []
    if (location / "current.json").exists():
        paths.append(location / read(location / "current.json")["path"])
    paths.extend(p.parent for p in sorted((location / "runs").glob("*/complete.json"), reverse=True))
    requested = reuse_identity(identity)
    for path in dict.fromkeys(paths):
        manifest = path / "complete.json"
        if not manifest.is_file():
            continue
        saved = read(manifest)["identity"]
        key = reuse_identity(saved)
        if key["gpu"] != requested["gpu"] or key["kernel_contract_version"] != requested["kernel_contract_version"]:
            continue
        if any(key["cases"].get(n) != value for n, value in requested["cases"].items()):
            continue
        # Compatible but corrupted records are errors, not permission to overwrite.
        return load_bundle(
            location, identity, reference=dict(path=str(path), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
        )[0]
    return None


def import_oracle(source, output, configs):
    oracle = load_oracle(source / "oracle.json")
    if {digest(c) for c in configs} != {digest(r["config"]) for r in oracle["records"].values()}:
        raise ValueError("Oracle pool mismatch")
    rows = sorted(oracle["records"].values(), key=lambda r: r["index"])
    tuning, work, evidence = [], [], []
    for attempt in sorted({r["attempt_path"] for r in rows}):
        path = Path(attempt)
        audit = read(path / "monitor.json")
        if audit.get("observed_foreign_processes"):
            raise ValueError(f"Contended oracle record: {attempt}")
        result = read(path / "result.json") if (path / "result.json").exists() else {}
        tuning.append(result.get("tuning_seconds"))
        work.append(audit["worker_seconds"])
        evidence.append(dict(path=str(path / "monitor.json"), sha256=hashlib.sha256((path / "monitor.json").read_bytes()).hexdigest()))
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "oracle.json", output / "oracle.json")
    write_json(output / "outcomes.json", rows)
    winner = oracle["winner"]
    result = dict(
        status="completed",
        correctness="passed",
        configs=len(rows),
        winner={k: winner[k] for k in ("index", "config", "latency_ms")},
        tuning_seconds=sum(tuning) if all(t is not None for t in tuning) else None,
        worker_seconds=sum(work),
        candidate_statuses=oracle["statuses"],
        timing_scope="sum of accepted shard invocation costs; not parallel elapsed time",
        contention_evidence=evidence,
    )
    write_json(output / "result.json", result)
    return result


def execute(args, plan):
    from experiments.common.comparison import training_sample
    from experiments.xgboost.data import read_runs
    from experiments.xgboost.model import train
    from experiments.compare_results import compare

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = replace(Device(**plan["devices"][0]), subsets=None)
    while not (
        available := [
            g for g in idle_gpus(snapshot()) if matches_cuda_device(g, device) and (not args.gpus or int(g["index"]) in args.gpus)
        ]
    ):
        time.sleep(2)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=available[0]["uuid"])
    runtime_path = output / "runtime.json"
    if not runtime_path.exists():
        code = "import json; from experiments.common.spec import Device; from experiments.utils.baseline_store import runtime_identity; print(json.dumps(runtime_identity(Device(**json.loads(__import__('sys').argv[1])))))"
        runtime = json.loads(
            subprocess.check_output([sys.executable, "-c", code, json.dumps(device.to_dict())], cwd=ROOT, env=env, text=True)
        )
        write_json(runtime_path, runtime)
    runtime = read(runtime_path)
    if (output / "plan.json").exists() and read(output / "plan.json") != plan:
        raise ValueError("Resume requires the original plan; use a new output for a different study")
    write_json(output / "plan.json", plan)
    timing = {k: SETTINGS[k] for k in ("workers", "warmup", "rep", "timeout", "case_timeout")}
    bundles, identities_by_family, missing, models = {}, {}, {}, {}
    for family in plan["families"]:
        fp = study_plan("full", [device], families=[family])
        measurement, identity = identities(fp, device, timing, runtime, 123)
        location = storage_root(family, device, runtime)
        existing = find_bundle(location, identity)
        if existing:
            bundles[family] = existing
            continue
        path = location / "runs" / ("cached-" + digest(reuse_identity(identity)))
        path.mkdir(parents=True, exist_ok=True)
        if (path / "identity.json").exists():
            identity = read(path / "identity.json")
        else:
            write_json(path / "identity.json", identity)
            write_json(path / "measurement.json", measurement)
        bundles[family], identities_by_family[family], missing[family] = path, identity, fp
    write_json(output / "cache-audit.json", dict(baselines={f: dict(path=str(p), reused=f not in missing) for f, p in bundles.items()}))

    if missing:
        workloads = [w for fp in missing.values() for w in fp["splits"]["test"]]
        manifest = output / "oracle-manifest.json"
        write_json(manifest, dict(version=1, devices=[device.to_dict()], workloads=workloads))
        oracle_root = output / "oracles"
        if not (oracle_root / "summary.json").exists():
            command = [
                sys.executable,
                "-m",
                "experiments.common.brute_force",
                "--device",
                device.name,
                "--manifest",
                str(manifest),
                "--output",
                str(oracle_root),
                "--workers",
                str(args.workers),
                "--shard-size",
                "32",
            ]
            if oracle_root.exists():
                command.append("--resume")
            if args.gpus:
                command.extend(["--gpus", *map(str, args.gpus)])
            with (output / "oracle.log").open("a") as stream:
                subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        training_jobs, paths = [], {}
        for family, fp in missing.items():
            collection = bundles[family] / "collection" / device.name
            paths[family] = {"train": [], "validation": []}
            models[family] = collection / "models" / ("xgboost-" + fp["splits"]["test"][0]["op"] + ".json")
            for split in paths[family]:
                for item in fp["splits"][split]:
                    w = Workload(**item)
                    path = collection / "xgboost" / split / w.name
                    paths[family][split].append(path)
                    training_jobs.append(
                        job("training-" + w.name, w, device, path, "brute_force", **training_sample(w, device, fraction=0.1, seed=123))
                    )
        run_jobs(training_jobs, output / "training-jobs", device, args.gpus)
        for family, split in paths.items():
            if not models[family].exists():
                train(
                    read_runs(split["train"]),
                    read_runs(split["validation"]),
                    models[family],
                    sample_fraction=0.1,
                    seed=123,
                    workers=args.workers,
                )
        baseline_jobs = []
        for family, fp in missing.items():
            for item in fp["splits"]["test"]:
                w = Workload(**item)
                case = bundles[family] / "collection" / device.name / "test" / w.name
                for method in ("carver", "xgboost"):
                    baseline_jobs.append(
                        job(
                            method + "-" + w.name,
                            w,
                            device,
                            case / method,
                            method,
                            **({"xgb_model": str(models[family])} if method == "xgboost" else {}),
                        )
                    )
        run_jobs(baseline_jobs, output / "baseline-jobs", device, args.gpus)
        for family, fp in missing.items():
            bundle, identity = bundles[family], identities_by_family[family]
            for item in fp["splits"]["test"]:
                w = Workload(**item)
                case = bundle / "collection" / device.name / "test" / w.name
                methods = {m: read(case / m / "result.json") for m in ("carver", "xgboost")}
                # Selections and the model were frozen without reading test labels.
                methods["brute_force"] = import_oracle(output / "oracles" / w.name, case / "brute_force", identity["pools"][w.name])
                write_json(case / "methods.json", methods)
                write_json(
                    case / "xgboost-model.json",
                    dict(path=str(models[family]), sha256=hashlib.sha256(models[family].read_bytes()).hexdigest()),
                )
                validate_case_bundle(case, identity["pools"][w.name])
            write_json(bundle / "complete.json", dict(version=1, identity=identity, artifacts=hash_files(bundle.rglob("*.json"), bundle)))
            verify_bundle(bundle, identity)
            publish_bundle(storage_root(family, device, runtime), bundle, identity)
    write_json(
        output / "baselines.json",
        {f: dict(path=str(p), manifest_sha256=hashlib.sha256((p / "complete.json").read_bytes()).hexdigest()) for f, p in bundles.items()},
    )

    profile_jobs, profile_dirs = [], {}
    gpu_name = runtime["device"].removeprefix("NVIDIA ")
    for item in plan["splits"]["test"]:
        w = Workload(**item)
        if w.dtype in profile_dirs:
            continue
        path = ROOT / "experiments/results/profiles" / gpu_name / w.dtype
        profile_dirs[w.dtype] = path
        if not successful_record(path):
            profile_jobs.append(job("profile-" + w.dtype, w, device, path, "profile"))
    run_jobs(profile_jobs, output / "profile-jobs", device, args.gpus)
    profiles = {dtype: read(path / "result.json")["profiles"][dtype] for dtype, path in profile_dirs.items()}
    write_json(
        output / "profiles.json",
        {
            dtype: dict(path=p, sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest(), result=str(profile_dirs[dtype] / "result.json"))
            for dtype, p in profiles.items()
        },
    )
    target = replace(device, profiles=profiles)
    revision = digest(hash_files([*ROOT.glob("tilelang/tiletune/**/*.py"), *ROOT.glob("tiletune_core/**/*.py")]))
    method_jobs, method_paths = [], {}
    for item in plan["splits"]["test"]:
        w = Workload(**item)
        family = FAMILIES[w.op]
        for seed in plan["budget"]["seeds"]:
            cache_key = digest(
                dict(
                    version=1,
                    workload=w.to_dict(),
                    configs=configuration_space(w, device)["configs"],
                    contract=2,
                    gpu=runtime["device"],
                    revision=revision,
                    profiles=read(output / "profiles.json"),
                    top_k=20,
                    seed=seed,
                )
            )
            path = ROOT / "experiments" / family / "results" / gpu_name / "tiletune/cache" / cache_key
            method_paths[w.name, seed] = path
            method_jobs.append(job(f"tiletune-{w.name}-{seed}", w, target, path, "top_k", selection_seed=seed))
    run_jobs(method_jobs, output / "tiletune-jobs", device, args.gpus)
    validation_jobs = []
    for item in plan["splits"]["test"]:
        w = Workload(**item)
        for seed in plan["budget"]["seeds"]:
            path = method_paths[w.name, seed]
            result = read(path / "result.json")
            if result["status"] == "completed":
                validation_jobs.append(
                    job(
                        f"validation-{w.name}-{seed}",
                        w,
                        target,
                        path / "winner-validation",
                        "remeasure",
                        methods={"tiletune": result},
                        validation_repeats=7,
                    )
                )
    run_jobs(validation_jobs, output / "validation-jobs", device, args.gpus)
    for item in plan["splits"]["test"]:
        w = Workload(**item)
        family = FAMILIES[w.op]
        baseline = bundles[family] / "collection" / device.name / "test" / w.name
        for seed in plan["budget"]["seeds"]:
            case = output / "comparison" / str(seed) / w.name
            case.mkdir(parents=True, exist_ok=True)
            paths = {m: baseline / m for m in ("brute_force", "carver", "xgboost")}
            paths["tiletune"] = method_paths[w.name, seed]
            for name, path in paths.items():
                link = case / name
                if not link.exists():
                    link.symlink_to(path, target_is_directory=True)
                elif link.resolve() != path.resolve():
                    raise ValueError(f"Comparison reference changed: {link}")
            curves = compare(paths["brute_force"], {m: paths[m] for m in ("carver", "xgboost", "tiletune")}, [1, 5, 10, 20, 50])
            write_json(case / "oracle-curves.json", curves)
            write_json(
                case / "comparison.json",
                dict(
                    workload=w.to_dict(),
                    seed=seed,
                    family=family,
                    baseline_bundle=str(bundles[family]),
                    methods={m: read(p / "result.json") for m, p in paths.items()},
                ),
            )
    from experiments.cached_report import report

    report(output)
    print(f"COMPLETED {output / 'report.md'}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("ampere", "hopper", "blackwell"), default="hopper")
    parser.add_argument("--families", nargs="+", choices=tuple(FAMILIES.values()), default=list(FAMILIES.values()))
    parser.add_argument("--gpus", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 456, 789])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    if args.job:
        return run_job(read(args.job))
    if args.output is None:
        parser.error("--output is required")
    if args.workers < 1 or any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("positive workers and distinct nonnegative seeds required")
    SETTINGS["workers"] = args.workers
    device = Device(args.device, TARGETS[args.device], expected_device_pattern=DEVICE_PATTERNS[args.device])
    plan = study_plan("full", [device], families=args.families)
    plan["budget"]["seeds"] = args.seeds
    if args.plan:
        print(json.dumps(plan, indent=2))
        return 0
    return execute(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
