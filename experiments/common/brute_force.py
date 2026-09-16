"""Measure final family pools on idle CUDA GPUs and export audited heuristics.

Run in the experiment's Python environment, from the repository root:
python -m experiments.common.brute_force --output experiments/results/h200-oracle
"""

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from experiments.utils.monitor import snapshot, foreign_processes, stop_worker
from .run import make_request, validate_result
from experiments.utils.io import write_json
from .spec import configuration_space, load_manifest, support_reason
from experiments.families import FAMILIES
from experiments.xgboost.data import canonical_workload


ROOT = Path(__file__).resolve().parents[2]
POISON = re.compile(
    r"illegal (memory access|address)|device.side assert|unspecified launch failure|CUDA_ERROR_(ILLEGAL|LAUNCH_FAILED|MISALIGNED)", re.I
)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def choose_winner(records):
    successful = [r for r in records if r["status"] == "benchmarked" and math.isfinite(r["latency_ms"]) and r["latency_ms"] > 0]
    if not successful:
        raise RuntimeError("No correct measured configuration in the complete pool")
    return min(successful, key=lambda r: (r["latency_ms"], r["index"]))


def export_case(root, workload, device, space, records, remeasurement, validation_dir, *, oracle_path=None):
    records = sorted(records.values(), key=lambda r: r["index"])
    if [r["index"] for r in records] != list(range(len(space["configs"]))) or [r["config"] for r in records] != space["configs"]:
        raise ValueError("heuristic export requires the complete ordered configuration pool")
    from experiments.utils.results import TERMINAL

    if any(r["status"] not in TERMINAL for r in records):
        raise ValueError("heuristic export requires terminal outcomes for every configuration")
    winner = choose_winner(records)
    attempt = Path(oracle_path).parent if oracle_path is not None else Path(winner["attempt_path"])
    experiment = json.loads((attempt / "experiment.json").read_text())
    measurement = json.loads((attempt / "result.json").read_text())["measurement"]
    observation = experiment["device_observation"]
    identity = dict(
        workload=canonical_workload(workload),
        target=device.target,
        device=observation["name"],
        kernel_sources=experiment["source_sha256"],
        native_build=experiment["native_build"],
        runtime={key: observation[key] for key in ("torch_version", "runtime_version", "device_compiler_version", "host_compiler_version")},
        measurement={**measurement, "input_seed": experiment["settings"]["seed"]},
    )
    if experiment.get("measurement_identity"):
        identity["measurement_identity"] = experiment["measurement_identity"]
    reference = Path(oracle_path) if oracle_path is not None else root / workload.name / "oracle.json"
    if oracle_path is None:
        for path in ("tilelang/profiler/__init__.py", "tilelang/profiler/bench.py"):
            identity["kernel_sources"][path] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        write_json(
            reference, dict(version=1, identity=identity, config_space=workload.config_space, candidate_count=len(records), records=records)
        )
    validation = remeasurement["validation"]["brute_force"]
    destination = (
        ROOT
        / "experiments"
        / FAMILIES[workload.op]
        / "heuristics"
        / observation["name"].removeprefix("NVIDIA ")
        / (workload.name + ".json")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        version=1,
        backend=observation["name"].removeprefix("NVIDIA "),
        recorded_at=timestamp(),
        workload=workload.to_dict(),
        device=observation["name"],
        target=device.target,
        config_space=workload.config_space,
        candidate_count=len(space["configs"]),
        outcomes=dict(Counter(r["status"] for r in records)),
        winner=dict(
            config=winner["config"],
            index=winner["index"],
            config_id=winner["config_id"],
            sweep_latency_ms=winner["latency_ms"],
            latency_ms=validation["median_ms"],
            samples_ms=validation["samples_ms"],
            relative_spread=validation["relative_spread"],
        ),
        reference=dict(path=str(reference), sha256=hashlib.sha256(reference.read_bytes()).hexdigest()),
        identity=identity,
        selection_rule="minimum correct latency in the complete measured configuration pool; ties use original index",
        contention=dict(
            poll_interval_seconds=1,
            policy="reject any invocation with an observed foreign compute process",
            observations_path=str((attempt if oracle_path is not None else root) / "gpu_observations.jsonl"),
            accepted_attempts_observed_uncontended=True,
            limitation="process polling cannot exclude interference shorter than the polling interval",
        ),
        validation_path=str(validation_dir),
        sweep_winner_gpu=winner.get("gpu") or json.loads((attempt / "monitor.json").read_text())["gpus"][0],
    )
    write_json(destination, result)
    return str(destination)


def export_baseline_study(study, output, device_name, workload_names=None):
    """Validate saved full baselines and remeasure winners without another sweep."""
    from .spec import Device, Workload
    from experiments.suite import _existing_or_run
    from experiments.utils.baseline_store import measurement_sources, runtime_identity, verify_bundle
    from experiments.utils.results import load_oracle

    lock = json.loads((study / "study-lock.json").read_text())
    plan, settings = lock["plan"], lock["settings"]
    if plan["suite"] not in ("full", "final"):
        raise ValueError("heuristic export requires a full or final study")
    device = Device(**next(d for d in plan["devices"] if d["name"] == device_name))
    workloads = [Workload(**w) for w in plan["splits"]["test"]]
    if workload_names:
        if set(workload_names) - {w.name for w in workloads}:
            raise ValueError("unknown study workload")
        workloads = [w for w in workloads if w.name in workload_names]
    references = json.loads((study / "baselines.json").read_text())[device_name]
    runtime = runtime_identity(device)
    bundles = {}
    # Verify everything before starting validation or publishing a heuristic.
    for op in dict.fromkeys(w.op for w in workloads):
        if references[op].get("status") == "unsupported":
            continue
        bundle = Path(references[op]["path"])
        if hashlib.sha256((bundle / "complete.json").read_bytes()).hexdigest() != references[op]["manifest_sha256"]:
            raise ValueError("baseline completion manifest changed")
        identity = json.loads((bundle / "identity.json").read_text())
        verify_bundle(bundle, identity)
        measurement = identity["measurement"]
        if measurement["runtime"] != runtime or measurement["sources"] != measurement_sources([FAMILIES[op]]):
            raise ValueError("baseline measurement environment changed; collect a new baseline before exporting")
        bundles[op] = bundle, measurement
    output.mkdir(parents=True, exist_ok=False)
    exported = {}
    unavailable = {}
    for w in workloads:
        if w.op not in bundles:
            unavailable[w.name] = references[w.op]
            continue
        bundle, measurement = bundles[w.op]
        oracle_path = bundle / "collection" / device_name / "test" / w.name / "brute_force/outcomes.json"
        monitor = json.loads((oracle_path.parent / "monitor.json").read_text())
        if monitor["status"] != "uncontended":
            raise ValueError("baseline oracle was not observed uncontended")
        oracle = load_oracle(oracle_path)
        records = {r["index"]: r for r in oracle["records"].values()}
        options = dict(
            settings,
            method="remeasure",
            metric="pipeline_time",
            top_k=1,
            seed=measurement["input_seed"],
            memory_regime="streaming",
            trace=False,
            measurement_identity=measurement,
            methods={"brute_force": dict(status="completed", winner=oracle["winner"])},
            validation_repeats=7,
        )
        directory = output / w.name / "winner-remeasurement"
        result = _existing_or_run(make_request(w, device, options), directory)
        if result["status"] != "remeasured":
            raise RuntimeError(f"Winner validation failed: {directory}: {result}")
        exported[w.name] = export_case(
            output, w, device, configuration_space(w, device), records, result, directory, oracle_path=oracle_path
        )
        write_json(output / "summary.json", dict(heuristic_files=exported, unavailable=unavailable, baseline_study=str(study)))
    write_json(output / "summary.json", dict(heuristic_files=exported, unavailable=unavailable, baseline_study=str(study)))
    return 0


def main():
    started_at, started = timestamp(), time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="hopper")
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/manifests/five_target_final.json")
    parser.add_argument("--workloads", nargs="+", help="Final workload names to measure; default: all cases in the manifest")
    parser.add_argument("--gpus", nargs="+", type=int)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--baseline-study", type=Path, help="Export a completed full study's saved oracles; only remeasure winners")
    args = parser.parse_args()
    if args.shard_size < 1 or args.workers < 1:
        parser.error("shard size and workers must be positive")
    if args.baseline_study:
        if args.resume or args.gpus:
            parser.error("baseline export uses the study device and a new output directory")
        return export_baseline_study(args.baseline_study.resolve(), args.output.resolve(), args.device, args.workloads)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    devices, workloads = load_manifest(json.loads(args.manifest.read_text()))
    if args.workloads:
        unknown = set(args.workloads) - {w.name for w in workloads}
        if unknown:
            parser.error(f"unknown final workloads: {sorted(unknown)}")
        workloads = [w for w in workloads if w.name in args.workloads]
    device = next(d for d in devices if d.name == args.device)
    if device.target["kind"] != "cuda":
        parser.error("this monitored runner requires CUDA")
    requested = workloads
    unavailable = {w.name: dict(status="unsupported", reason=reason) for w in requested if (reason := support_reason(w, device))}
    workloads = [w for w in requested if w.name not in unavailable]
    initial = snapshot()
    gpus = [
        g
        for g in initial["gpus"]
        if (args.gpus is None or int(g["index"]) in args.gpus) and re.search(device.expected_device_pattern, g["name"])
    ]
    if not gpus:
        parser.error("no matching GPUs")
    spaces = {w.name: configuration_space(w, device) for w in workloads}
    settings = dict(
        method="brute_force",
        metric="pipeline_time",
        top_k=1,
        memory_regime="streaming",
        trace=False,
        workers=args.workers,
        warmup=10,
        rep=50,
        timeout=30,
        case_timeout=1800,
        seed=123,
    )
    plan = dict(
        version=1,
        workloads=[w.to_dict() for w in requested],
        unavailable=unavailable,
        device=device.to_dict(),
        settings=settings,
        shard_size=args.shard_size,
        config_ids={name: space["config_ids"] for name, space in spaces.items()},
    )
    plan_path = root / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Plan changed; use a new output directory")
    write_json(plan_path, plan)
    for name, result in unavailable.items():
        (root / name).mkdir(exist_ok=True)
        write_json(root / name / "result.json", result)
    from experiments.utils.cli import source_hashes
    from tilelang.cache.kernel_cache import KernelCache

    provenance = dict(source_sha256=source_hashes("experiments/common/kernels.py"), native_build=KernelCache._get_tilelang_lib_stamp())
    if (root / "provenance.json").exists() and json.loads((root / "provenance.json").read_text()) != provenance:
        raise ValueError("Source or native build changed; use a new output directory")
    write_json(root / "provenance.json", provenance)
    records, pending, active, exported = {}, deque(), {}, {}
    for w in workloads:
        directory = root / w.name
        directory.mkdir(exist_ok=True)
        write_json(directory / "config-space.json", spaces[w.name])
        records[w.name] = {}
        for path in sorted(directory.glob("attempt-*/accepted.json")):
            for row in json.loads(path.read_text()):
                if row["index"] in records[w.name]:
                    raise ValueError("duplicate accepted configuration")
                records[w.name][row["index"]] = row
    # Round-robin workloads, then dynamically share their remaining shards.
    for offset in range(0, max((len(s["configs"]) for s in spaces.values()), default=0), args.shard_size):
        for w in workloads:
            indices = [i for i in range(offset, min(offset + args.shard_size, len(spaces[w.name]["configs"]))) if i not in records[w.name]]
            if indices:
                pending.append(dict(workload=w, indices=indices, kind="sweep"))
    clean = {g["uuid"]: 0 for g in gpus}
    validating = set()
    last_progress = 0

    def event(value):
        with (root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(timestamp=timestamp(), **value)) + "\n")
        print(json.dumps(value), flush=True)

    try:
        with (root / "gpu_observations.jsonl").open("a") as monitor:
            while pending or active or len(exported) < len(workloads):
                observed = snapshot()
                observed["workers"] = {
                    uuid: dict(pid=a["process"].pid, workload=a["job"]["workload"].name, attempt=str(a["directory"]))
                    for uuid, a in active.items()
                }
                monitor.write(json.dumps(observed) + "\n")
                monitor.flush()
                for gpu in gpus:
                    uuid = gpu["uuid"]
                    task = active.get(uuid)
                    foreign = foreign_processes(observed, uuid, task["process"].pid if task else None)
                    gpu_now = next(g for g in observed["gpus"] if g["uuid"] == uuid)
                    clean[uuid] = clean[uuid] + 1 if not foreign and (task or float(gpu_now["utilization.gpu"]) <= 5) else 0
                    if task:
                        process, job, directory = task["process"], task["job"], task["directory"]
                        timeout = time.monotonic() - task["started"] > settings["case_timeout"]
                        if foreign or timeout:
                            stop_worker(process)
                        if process.poll() is None:
                            continue
                        task["log"].close()
                        del active[uuid]
                        clean[uuid] = 0
                        w = job["workload"]
                        audit = dict(
                            started_at=task["started_at"],
                            ended_at=timestamp(),
                            gpu=gpu,
                            worker_pid=process.pid,
                            observed_foreign_processes=foreign,
                            timeout=timeout,
                            exit_code=process.returncode,
                            worker_seconds=time.monotonic() - task["started"],
                        )
                        write_json(directory / "monitor.json", audit)
                        if foreign:
                            pending.appendleft(job)
                            event(dict(event="discard_contended", workload=w.name, gpu=gpu["index"], foreign=foreign))
                            continue
                        result_path = directory / "result.json"
                        result = (
                            json.loads(result_path.read_text())
                            if result_path.exists()
                            else dict(status="failed", reason=f"worker exit {process.returncode}, timeout={timeout}")
                        )
                        if result_path.exists():
                            validate_result(result, task["request"])
                        if job["kind"] == "validation":
                            if result["status"] != "remeasured":
                                raise RuntimeError(f"Winner validation failed: {directory}: {result}")
                            exported[w.name] = export_case(root, w, device, spaces[w.name], records[w.name], result, directory)
                            event(dict(event="exported", workload=w.name, path=exported[w.name]))
                            continue
                        outcomes_path = directory / "outcomes.json"
                        rows = json.loads(outcomes_path.read_text()) if outcomes_path.exists() else []
                        if not rows and not (directory / "experiment.json").exists():
                            raise RuntimeError(f"Worker failed before candidate execution: {directory}: {result}")
                        accepted, retry = [], []
                        by_index = {r["original_index"]: r for r in rows}
                        for i in job["indices"]:
                            row = by_index.get(i)
                            incomplete = row is None or row["status"] in ("selected", "compiled", "not_attempted")
                            poisoned = row is not None and POISON.search(row.get("error") or "")
                            if (incomplete or poisoned) and len(job["indices"]) > 1:
                                retry.append(i)
                                continue
                            if row is None or incomplete:
                                row = dict(
                                    config=spaces[w.name]["configs"][i],
                                    config_id=spaces[w.name]["config_ids"][i],
                                    status="worker_failed",
                                    error=result.get("reason"),
                                    latency_ms=None,
                                )
                            row.update(index=i, original_index=i, gpu=gpu, attempt_path=str(directory), observed_uncontended=True)
                            if i in records[w.name]:
                                raise ValueError("duplicate accepted configuration")
                            records[w.name][i] = row
                            accepted.append(row)
                        write_json(directory / "accepted.json", accepted)
                        write_json(root / w.name / "outcomes.json", sorted(records[w.name].values(), key=lambda r: r["index"]))
                        if retry:
                            half = max(1, len(retry) // 2)
                            for offset in range(0, len(retry), half):
                                pending.append(dict(workload=w, indices=retry[offset : offset + half], kind="sweep"))
                        event(
                            dict(
                                event="shard_finished",
                                workload=w.name,
                                gpu=gpu["index"],
                                completed=len(records[w.name]),
                                total=len(spaces[w.name]["configs"]),
                                retry=len(retry),
                                outcomes=dict(Counter(r["status"] for r in accepted)),
                            )
                        )
                    if uuid not in active and clean[uuid] >= 3 and pending:
                        job = pending.popleft()
                        w = job["workload"]
                        directory = root / w.name / ("attempt-" + str(time.time_ns()))
                        directory.mkdir()
                        request_settings = (
                            dict(settings, config_indices=job["indices"])
                            if job["kind"] == "sweep"
                            else dict(
                                settings,
                                method="remeasure",
                                methods={"brute_force": dict(status="completed", winner=choose_winner(list(records[w.name].values())))},
                                validation_repeats=7,
                            )
                        )
                        request = make_request(w, device, request_settings)
                        write_json(directory / "request.json", request)
                        env = dict(
                            os.environ,
                            CUDA_DEVICE_ORDER="PCI_BUS_ID",
                            CUDA_VISIBLE_DEVICES=uuid,
                            OMP_NUM_THREADS="1",
                            MKL_NUM_THREADS="1",
                            OPENBLAS_NUM_THREADS="1",
                        )
                        log = (directory / "worker.log").open("w")
                        process = subprocess.Popen(
                            [
                                sys.executable,
                                "-m",
                                "experiments.common.run",
                                "--worker",
                                str(directory / "request.json"),
                                str(directory / "result.json"),
                            ],
                            cwd=ROOT,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        active[uuid] = dict(
                            process=process,
                            log=log,
                            job=job,
                            directory=directory,
                            request=request,
                            started=time.monotonic(),
                            started_at=timestamp(),
                        )
                        event(
                            dict(
                                event="started",
                                workload=w.name,
                                kind=job["kind"],
                                count=len(job["indices"]),
                                gpu=gpu["index"],
                                pid=process.pid,
                            )
                        )
                for w in workloads:
                    if len(records[w.name]) == len(spaces[w.name]["configs"]) and w.name not in validating:
                        choose_winner(list(records[w.name].values()))
                        pending.append(dict(workload=w, indices=[], kind="validation"))
                        validating.add(w.name)
                if time.monotonic() - last_progress >= 30:
                    progress = dict(
                        timestamp=timestamp(),
                        completed={w.name: len(records[w.name]) for w in workloads},
                        total={w.name: len(spaces[w.name]["configs"]) for w in workloads},
                        active=len(active),
                        queued=len(pending),
                        exported=exported,
                    )
                    write_json(root / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
                    last_progress = time.monotonic()
                time.sleep(1)
    finally:
        for task in active.values():
            stop_worker(task["process"])
            task["log"].close()
    write_json(
        root / "summary.json",
        dict(
            started_at=started_at,
            completed_at=timestamp(),
            invocation_wall_seconds=time.monotonic() - started,
            heuristic_files=exported,
            unavailable=unavailable,
            outcomes={w.name: dict(Counter(r["status"] for r in records[w.name].values())) for w in workloads},
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
