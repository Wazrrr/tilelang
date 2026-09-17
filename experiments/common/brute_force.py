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

from experiments.utils.monitor import snapshot, foreign_processes, stop_worker, visible_gpus, matches_cuda_device
from .run import make_request, validate_result
from experiments.utils.io import write_json
from .spec import configuration_space, load_manifest
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


def export_case(root, workload, device, space, records, remeasurement, validation_dir):
    records = sorted(records.values(), key=lambda r: r["index"])
    winner = choose_winner(records)
    experiment = json.loads((Path(winner["attempt_path"]) / "experiment.json").read_text())
    observation = experiment["device_observation"]
    identity = dict(
        workload=canonical_workload(workload),
        target=device.target,
        device=observation["name"],
        kernel_sources=experiment["source_sha256"],
        native_build=experiment["native_build"],
        runtime={key: observation[key] for key in ("torch_version", "runtime_version", "device_compiler_version", "host_compiler_version")},
        measurement=dict(backend="event", memory_regime="streaming", cache_flush_bytes=268435456, input_seed=123, warmup=10, rep=50),
    )
    for path in ("tilelang/profiler/__init__.py", "tilelang/profiler/bench.py"):
        identity["kernel_sources"][path] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    reference = root / workload.name / "oracle.json"
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
            policy="discard and retry any shard with an observed foreign compute process",
            observations_path=str(root / "gpu_observations.jsonl"),
            accepted_attempts_observed_uncontended=True,
            limitation="process polling cannot exclude interference shorter than the polling interval",
        ),
        validation_path=str(validation_dir),
        sweep_winner_gpu=winner["gpu"],
    )
    write_json(destination, result)
    return str(destination)


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
    args = parser.parse_args()
    if args.shard_size < 1 or args.workers < 1:
        parser.error("shard size and workers must be positive")
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
    initial = snapshot()
    gpus = [g for g in visible_gpus(initial) if (args.gpus is None or int(g["index"]) in args.gpus) and matches_cuda_device(g, device)]
    if not gpus:
        parser.error("no matching GPUs")
    if args.gpus is not None and len(gpus) != len(set(args.gpus)):
        parser.error("requested GPUs must be visible and match the target")
    if args.gpus is None:
        gpus = [g for g in gpus if g["name"] == gpus[0]["name"]]
    if len({g["name"] for g in gpus}) != 1:
        parser.error("select GPUs of one model; baseline measurements must not mix models")
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
        workloads=[w.to_dict() for w in workloads],
        device=device.to_dict(),
        settings=settings,
        shard_size=args.shard_size,
        config_ids={name: space["config_ids"] for name, space in spaces.items()},
    )
    plan_path = root / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Plan changed; use a new output directory")
    write_json(plan_path, plan)
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
    for offset in range(0, max(len(s["configs"]) for s in spaces.values()), args.shard_size):
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
            outcomes={w.name: dict(Counter(r["status"] for r in records[w.name].values())) for w in workloads},
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
