"""Sequential, resumable H200 E1/E2/E3 study on one fixed set of four GPUs.

python -m experiments.common.h200 --plan
python -m experiments.common.h200 --gpus 0 1 2 3 --output experiments/results/h200-study
python -m experiments.common.h200 --gpus 0 1 2 3 --output experiments/results/h200-study --resume
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from experiments.common.spaces import config_id
from experiments.common.spec import default_workloads
from experiments.families import family_module
from experiments.utils.io import write_json


MODES = {"E1": "baseline", "E2": "multi_gpu", "E3": "tiletune"}
SETTINGS = dict(workers=128, warmup=10, rep=50, timeout=60, group_size=8, seed=123)
RETRYABLE = {"contended", "host_contended", "monitor_gap"}


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    from experiments.utils.cli import source_hashes
    from experiments.utils.baseline_store import measurement_sources
    from experiments.families import FAMILIES

    root = Path(__file__).resolve().parents[2]
    sources = {**source_hashes("experiments/common/h200.py"), **measurement_sources(set(FAMILIES.values()))}
    # The worktree reuses a native build: a rebuild must invalidate resumption.
    native = {
        str(p.relative_to(root)): dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
        for directory in (root / "build/lib", root / "build/tvm")
        for p in directory.glob("*.so")
    }
    return dict(sources=sources, native_build=native)


def study_plan(*, workloads=None, experiments=None, preflight=False):
    if experiments is not None and (not experiments or len(experiments) != len(set(experiments)) or set(experiments) - MODES.keys()):
        raise ValueError("experiments must be distinct E1/E2/E3 names")
    cases = default_workloads(smoke=False)
    if workloads:
        if set(workloads) - {w.name for w in cases}:
            raise ValueError("unknown final workload")
        cases = [w for w in cases if w.name in workloads]
    elif preflight:
        cases = list({w.op: next(c for c in cases if c.op == w.op) for w in cases}.values())
    rows = []
    for experiment in experiments or MODES:
        for workload in cases:
            pool = family_module(workload.op, "spaces").get_configs()
            indices = list(range(len(pool)))
            if preflight and experiment != "E3":
                indices = indices[:8]
            rows.append(
                dict(
                    experiment=experiment,
                    variant=MODES[experiment],
                    workload=workload.to_dict(),
                    indices=indices,
                    configs=[pool[i] for i in indices],
                    gpu_count=1 if experiment == "E1" else 4,
                    settings=dict(SETTINGS, preflight=preflight),
                )
            )
    return rows


def select_gpus(observation, requested=None):
    from experiments.utils.monitor import visible_gpus

    visible = visible_gpus(observation)
    if requested is not None:
        if len(requested) != 4 or len(set(requested)) != 4:
            raise ValueError("the study requires exactly four distinct GPU indices")
        by_index = {int(g["index"]): g for g in visible}
        if set(requested) - by_index.keys():
            raise ValueError("requested GPUs are not all visible")
        selected = [by_index[i] for i in requested]
    else:
        selected = [g for g in visible if "H200" in g["name"]][:4]
    if len(selected) != 4 or len({g["name"] for g in selected}) != 1 or any("H200" not in g["name"] for g in selected):
        raise ValueError("four matching H200 GPUs are required")
    return [{key: gpu[key] for key in ("index", "uuid", "name", "compute_cap")} for gpu in selected]


def validate_attempt(output, request):
    """A worker summary alone never establishes a completed, uncontended sweep."""
    if read(output / "monitor.json")["status"] != "uncontended":
        raise ValueError("attempt has no uncontended monitor completion")
    saved_request = read(output / "request.json")
    if any(saved_request.get(k) != v for k, v in request.items()):
        raise ValueError("attempt request identity mismatch")
    summary = read(output / "summary.json")
    if summary.get("status") != "completed" or summary.get("config_count") != len(request["configs"]):
        raise ValueError("incomplete worker summary")
    if summary.get("compiler_workers") != 128 or summary.get("benchmark_gpu_count") != request["gpu_count"]:
        raise ValueError("worker did not use the frozen CPU/GPU counts")
    if request.get("source_identity") and read(output / "experiment.json").get("source_identity") != request["source_identity"]:
        raise ValueError("worker source identity changed")
    records = read(output / "outcomes.json")
    from experiments.utils.results import TERMINAL

    allowed = TERMINAL | ({"not_selected"} if request["experiment"] == "E3" else set())
    if request["settings"]["preflight"]:
        allowed |= {"preflight_omitted"}
    if len(records) != len(request["configs"]):
        raise ValueError("incomplete candidate outcomes")
    for index, (record, config) in enumerate(zip(records, request["configs"])):
        if (
            record.get("index") != index
            or record.get("config") != config
            or record.get("config_id") != config_id(config)
            or record.get("status") not in allowed
        ):
            raise ValueError(f"invalid or nonterminal outcome at index {index}")
        if record["status"] == "benchmarked" and (
            not isinstance(record.get("latency_ms"), int | float) or not math.isfinite(record["latency_ms"]) or record["latency_ms"] <= 0
        ):
            raise ValueError("successful candidate requires a finite positive latency")
    winner = summary.get("winner_config")
    if not any(
        r["config"] == winner and r["status"] == "benchmarked" and r["latency_ms"] == summary.get("winner_latency_ms") for r in records
    ):
        raise ValueError("winner is absent from successful candidate measurements")
    if request["experiment"] == "E3":
        report = read(output / "tiletune.json")
        selection = report["selection"]
        selected = selection["selected_indices"]
        if (
            selection.get("alpha") != 0.5
            or not selection.get("strict_budget")
            or selection.get("pool_size") != len(records)
            or len(selected) > len(records) // 2
            or len(selected) != len(set(selected))
            or selection != read(output / "selection.json")
        ):
            raise ValueError("TileTune selection violates the frozen alpha contract")
    return summary


def completed_attempt(case_dir, request):
    marker = case_dir / "completed.json"
    if not marker.exists():
        return None
    completion = read(marker)
    if completion["request_id"] != digest(request):
        raise ValueError("completed workload belongs to a different request")
    output = case_dir / completion["attempt"]
    for name, expected in completion["files"].items():
        if file_hash(output / name) != expected:
            raise ValueError(f"completed artifact changed: {output / name}")
    validate_attempt(output, request)
    return output


def run_queue(root, plan, gpus, cpu_ids, lease_fds, *, run=None, max_contention_retries=3, code_identity=None):
    from experiments.utils.monitor import run_monitored

    run = run or run_monitored
    rows = []
    for item in plan:
        request = dict(item, cpu_ids=cpu_ids, source_identity=code_identity)
        case_dir = root / item["experiment"] / item["workload"]["name"]
        case_dir.mkdir(parents=True, exist_ok=True)
        output = completed_attempt(case_dir, request)
        retries = 0
        while output is None:
            number = max([int(p.name.split("-")[1]) for p in case_dir.glob("attempt-*") if p.is_dir()] or [0]) + 1
            output = case_dir / f"attempt-{number:04d}"
            output.mkdir()
            active = gpus[: item["gpu_count"]]
            write_json(output / "request.json", dict(request, parent_pid=os.getpid()))
            write_json(
                root / "progress.json",
                dict(
                    status="running",
                    completed=len(rows),
                    total=len(plan),
                    active=dict(experiment=item["experiment"], workload=item["workload"]["name"], attempt=str(output)),
                ),
            )
            write_json(output / "attempt.json", dict(status="running", request_id=digest(request)))
            temporary = output / "tmp"
            temporary.mkdir()
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=",".join(g["uuid"] for g in active),
                TILELANG_DISABLE_CACHE="1",
                TILELANG_AUTO_TUNING_DISABLE_CACHE="1",
                TILELANG_AUTO_TUNING_CPU_COUNTS="128",
                TILELANG_AUTO_TUNING_MAX_CPU_COUNT="128",
                TILELANG_AUTOTUNE_TIMING_LOG=str(output / "timings.tsv"),
                TMPDIR=str(temporary),
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                NUMEXPR_NUM_THREADS="1",
                TVM_NUM_THREADS="1",
            )
            print(f"{item['experiment']} {item['workload']['name']}: attempt {number}, {len(active)} GPU(s)", flush=True)
            try:
                run(
                    [sys.executable, "-m", "experiments.common.system", "--worker", str(output / "request.json"), str(output)],
                    output,
                    active,
                    env=env,
                    cwd=Path(__file__).resolve().parents[2],
                    cpu_ids=cpu_ids,
                    pass_fds=lease_fds,
                )
                summary = validate_attempt(output, request)
            except BaseException as error:
                status = read(output / "monitor.json").get("status") if (output / "monitor.json").exists() else "failed"
                status = "interrupted" if isinstance(error, KeyboardInterrupt) else status
                if status not in RETRYABLE | {"interrupted", "timeout", "worker_failed"}:
                    status = "failed"
                write_json(output / "attempt.json", dict(status=status, error=str(error), request_id=digest(request)))
                write_json(
                    root / "progress.json",
                    dict(status=status, completed=len(rows), total=len(plan), active=None, last_attempt=str(output), error=str(error)),
                )
                if status in RETRYABLE and retries < max_contention_retries:
                    retries += 1
                    print(f"Discarding contended attempt; waiting for quiet resources before retry {retries}.", flush=True)
                    output = None
                    continue
                raise
            files = [
                "request.json",
                "summary.json",
                "monitor.json",
                "outcomes.json",
                "experiment.json",
                "compilation.json",
                "benchmarks.tsv",
            ]
            if item["experiment"] == "E3":
                files += ["tiletune.json", "selection.json"]
            write_json(output / "attempt.json", dict(status="completed", request_id=digest(request)))
            write_json(
                case_dir / "completed.json",
                dict(request_id=digest(request), attempt=output.name, files={name: file_hash(output / name) for name in files}),
            )
        summary = read(output / "summary.json")
        row = dict(
            summary,
            experiment=item["experiment"],
            attempt=str(output.relative_to(root)),
            worker_wall_seconds=read(output / "monitor.json").get("wall_seconds"),
        )
        rows.append(row)
        baseline = next((r for r in rows if r["workload"] == row["workload"] and r["experiment"] == "E1"), None)
        if baseline is not None:
            row["tuning_speedup_vs_E1"] = baseline["tuning_seconds"] / row["tuning_seconds"]
            row["end_to_end_speedup_vs_E1"] = baseline["worker_wall_seconds"] / row["worker_wall_seconds"]
        write_json(root / "comparison.json", rows)
        write_json(
            root / "progress.json",
            dict(status="running" if len(rows) < len(plan) else "completed", completed=len(rows), total=len(plan), active=None),
        )
    return rows


def oracle_audit(root, plan):
    """Retain every exact E1/E2 minimum and audit all four E3 retention stages."""
    if any(item["settings"]["preflight"] for item in plan):
        return None
    by_name = {}
    for item in plan:
        path = root / item["experiment"] / item["workload"]["name"]
        marker = read(path / "completed.json")
        by_name.setdefault(item["workload"]["name"], {})[item["experiment"]] = path / marker["attempt"]
    rows = []
    for name, modes in by_name.items():
        if set(modes) != set(MODES):
            continue
        report = read(modes["E3"] / "tiletune.json")
        records = read(modes["E3"] / "outcomes.json")
        ranked = {r["index"]: r for r in report["ranking"]}
        selected = set(report["selection"]["selected_indices"])
        for baseline in ("E1", "E2"):
            pool = read(modes[baseline] / "outcomes.json")
            if [r["config_id"] for r in pool] != [r["config_id"] for r in records]:
                raise ValueError("oracle and TileTune pool identities differ")
            valid = [r for r in pool if r["status"] == "benchmarked" and r.get("latency_ms", 0) > 0]
            best = min(r["latency_ms"] for r in valid)
            for oracle in (r for r in valid if r["latency_ms"] == best):
                index = oracle["index"]
                rank, record = ranked[index], records[index]
                scored = isinstance(rank.get("score"), int | float) and math.isfinite(rank["score"]) and rank["tier"] == "eligible"
                tail = rank.get("tie_last_rank")
                rank_pass = scored and tail is not None and tail <= len(pool) // 2
                post_pass = (record.get("post_compile") or {}).get("status") == "pass"
                usable = record["status"] == "benchmarked"
                rows.append(
                    dict(
                        workload=name,
                        baseline=baseline,
                        config_id=oracle["config_id"],
                        config=oracle["config"],
                        oracle_latency_ms=best,
                        pool_size=len(pool),
                        alpha_budget=len(pool) // 2,
                        memory_score=rank.get("score"),
                        tail_rank=tail,
                        tail_fraction=tail / len(pool) if tail is not None else None,
                        rank_pass=rank_pass,
                        selected=index in selected,
                        post_compile_pass=post_pass,
                        correctness_and_benchmark_pass=usable,
                        post_compile=record.get("post_compile"),
                        status=record["status"],
                        retained=rank_pass and index in selected and post_pass and usable,
                    )
                )
    names = {row["workload"] for row in rows}
    result = dict(
        workloads_audited=len(names),
        workloads_retained=sum(all(r["retained"] for r in rows if r["workload"] == n) for n in names),
        required_workloads=len(by_name),
        complete=len(names) == len(by_name),
        oracles=rows,
    )
    write_json(root / "oracle_retention.json", result)
    if rows:
        with (root / "oracle_retention.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="separate 15-run preflight, at most eight compiled candidates per mode")
    parser.add_argument("--workloads", nargs="+")
    parser.add_argument("--experiments", choices=list(MODES), nargs="+")
    args = parser.parse_args(argv)
    plan = study_plan(workloads=args.workloads, experiments=args.experiments, preflight=args.preflight)
    if args.plan:
        print(json.dumps(dict(plan=plan, max_gpus=4, concurrent_workloads=1), indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required")
    from experiments.utils.isolation import measurement_lease, interruptible, select_cpu_ids
    from experiments.utils.monitor import snapshot

    root = args.output.resolve()
    if root.exists() and not args.resume:
        raise FileExistsError("output already exists; use --resume to verify and continue it")
    if args.resume and not (root / "manifest.json").exists():
        raise ValueError("resume requires an existing frozen manifest")
    frozen = read(root / "manifest.json") if args.resume else None
    observed = snapshot()
    requested = args.gpus if args.gpus is not None else [int(g["index"]) for g in frozen["gpus"]] if frozen else None
    gpus = select_gpus(observed, requested)
    identity = dict(
        version=1,
        plan=plan,
        gpus=gpus,
        source_identity=source_identity(),
        python=str(Path(sys.executable).resolve()),
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    if frozen and any(frozen.get(key) != value for key, value in identity.items()):
        raise ValueError("resume identity changed: workload, pool, code, settings, interpreter or GPUs differ")
    with measurement_lease(gpus) as leases, interruptible():
        cpu_ids = frozen["cpu_ids"] if frozen else select_cpu_ids(128)
        if not set(cpu_ids) <= os.sched_getaffinity(0):
            raise RuntimeError("frozen CPU affinity is no longer available")
        root.mkdir(parents=True, exist_ok=args.resume)
        if not frozen:
            write_json(root / "manifest.json", dict(identity, cpu_ids=cpu_ids))
        # The coordinator and its monitor do not compete with compilation threads.
        affinity = os.sched_getaffinity(0)
        spare = affinity - set(cpu_ids)
        if not spare:
            raise RuntimeError("no CPU left for the measurement coordinator")
        os.sched_setaffinity(0, spare)
        try:
            if not args.preflight:
                preflight_root = root / "preflight"
                preflight_root.mkdir(exist_ok=True)
                preflight = study_plan(experiments=args.experiments, preflight=True)
                write_json(root / "progress.json", dict(status="preflight", total=len(plan), completed=0))
                run_queue(preflight_root, preflight, gpus, cpu_ids, leases, code_identity=identity["source_identity"])
            run_queue(root, plan, gpus, cpu_ids, leases, code_identity=identity["source_identity"])
            audit = oracle_audit(root, plan)
            if audit is not None:
                print(json.dumps({k: v for k, v in audit.items() if k != "oracles"}), flush=True)
        finally:
            os.sched_setaffinity(0, affinity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
