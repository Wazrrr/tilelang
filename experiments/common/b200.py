"""Sequential, resumable B200 E1/E2/E3 study.

python -m experiments.common.b200 --plan
python -m experiments.common.b200 --experiments E1 --output experiments/results/b200-study/E1 --resume-or-start
python -m experiments.common.b200 --combine experiments/results/b200-study
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
from experiments.common.spec import Workload, default_workloads
from experiments.families import FAMILIES, family_module
from experiments.utils.io import write_json


MODES = {"E1": "baseline", "E2": "multi_gpu", "E3": "tiletune"}
COMPILER_WORKERS = 64
CPU_HEADROOM = 8
SETTINGS = dict(
    workers=COMPILER_WORKERS,
    warmup=10,
    rep=50,
    timeout=60,
    group_size=8,
    seed=123,
    benchmark_backend="event",
)
EXHAUSTIVE_RESOURCE_POLICY = "report"
# E1/E2 establish the unfiltered oracle resource requirements.  E3 then uses
# the family-calibrated limits in resource_policy.py as a strict gate.
TILETUNE_RESOURCE_POLICY = "reject"
RETRYABLE = {"contended", "host_contended", "monitor_gap"}
SLURM_METADATA = (
    "SLURM_CLUSTER_NAME",
    "SLURM_JOB_ID",
    "SLURM_JOB_NODELIST",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_QOS",
    "SLURM_RESTART_COUNT",
    "SLURMD_NODENAME",
)
ORCHESTRATION_SOURCES = {
    "experiments/common/b200.py",
    "experiments/common/run_b200_slurm.sh",
}
REUSE_POLICY_SOURCES = {
    "experiments/common/resource_policy.py",
    "experiments/common/system.py",
    "tilelang/tiletune/config.py",
    "tilelang/tiletune/runtime.py",
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def slurm_metadata():
    return {key.lower(): os.environ[key] for key in SLURM_METADATA if key in os.environ}


def measurement_identity(identity):
    """Exclude launch orchestration from the code that defines a measurement."""
    identity = dict(identity or {})
    identity["sources"] = {
        path: value for path, value in identity.get("sources", {}).items() if path not in ORCHESTRATION_SOURCES
    }
    return identity


def compatible_source_identity(left, right):
    return measurement_identity(left) == measurement_identity(right)


def compatible_reuse_source_identity(left, right):
    """Allow only the audited E1/E2 resource-policy worker change."""
    left, right = measurement_identity(left), measurement_identity(right)
    left_sources = dict(left.get("sources", {}))
    right_sources = dict(right.get("sources", {}))
    for path in REUSE_POLICY_SOURCES:
        left_sources.pop(path, None)
        right_sources.pop(path, None)
    return left_sources == right_sources and left.get("native_build", {}) == right.get("native_build", {})


def gpu_hardware_signature(gpus):
    """Describe interchangeable hardware without allocation-local indices or UUIDs."""
    return sorted((gpu.get("name"), gpu.get("compute_cap")) for gpu in gpus)


def validate_resume_identity(frozen, current):
    for key in ("version", "plan", "python", "revision", "reuse"):
        if frozen.get(key) != current.get(key):
            raise ValueError(f"resume identity changed: {key} differs")
    if not compatible_source_identity(frozen.get("source_identity", {}), current.get("source_identity", {})):
        raise ValueError("resume identity changed: measurement code or native build differs")
    if gpu_hardware_signature(frozen.get("gpus", [])) != gpu_hardware_signature(current.get("gpus", [])):
        raise ValueError("resume identity changed: GPU count, model or compute capability differs")


def compatible_request(saved, current):
    saved, current = dict(saved), dict(current)
    saved_source = saved.pop("source_identity", {})
    current_source = current.pop("source_identity", {})
    saved_cpus = saved.pop("cpu_ids", None)
    current_cpus = current.pop("cpu_ids", None)
    cpu_compatible = (
        saved_cpus == current_cpus
        if saved_cpus is None or current_cpus is None
        else len(saved_cpus) == len(current_cpus)
    )
    return saved == current and cpu_compatible and compatible_source_identity(saved_source, current_source)


def reuse_key(item):
    return f"{item['experiment']}/{item['workload']['name']}"


def prepare_reuse_specs(reuse_root, plan, code_identity, gpus):
    """Validate immutable completed E1/E2 attempts and freeze their reuse map."""
    reuse_root = Path(reuse_root).resolve()
    manifest_path = reuse_root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("--reuse-from must name a mode root containing manifest.json")
    old_manifest = read(manifest_path)
    if not compatible_reuse_source_identity(old_manifest.get("source_identity", {}), code_identity):
        raise ValueError("reuse source differs beyond the audited post-compile resource-policy change")
    if old_manifest.get("revision") != subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip():
        raise ValueError("reuse source revision differs")

    from experiments.utils.results import TERMINAL

    specs = {}
    required_files = {
        "request.json",
        "summary.json",
        "monitor.json",
        "outcomes.json",
        "experiment.json",
        "compilation.json",
        "benchmarks.tsv",
        "resource-filter.tsv",
    }
    for item in plan:
        if item["experiment"] not in ("E1", "E2") or item["settings"].get("preflight"):
            raise ValueError("reuse is supported only for full E1/E2 sweeps")
        case_dir = reuse_root / item["experiment"] / item["workload"]["name"]
        marker = case_dir / "completed.json"
        if not marker.is_file():
            continue
        completion = read(marker)
        output = case_dir / completion["attempt"]
        files = completion.get("files", {})
        if not required_files <= files.keys():
            raise ValueError(f"reuse attempt lacks required artifacts: {output}")
        for name, expected in files.items():
            if file_hash(output / name) != expected:
                raise ValueError(f"reuse artifact changed: {output / name}")

        saved_request = read(output / "request.json")
        saved_identity = {key: value for key, value in saved_request.items() if key != "parent_pid"}
        if completion.get("request_id") != digest(saved_identity):
            raise ValueError(f"reuse request identity mismatch: {output}")
        for key in ("experiment", "variant", "workload", "indices", "configs", "gpu_count"):
            if saved_identity.get(key) != item[key]:
                raise ValueError(f"reuse request changed {key}: {output}")
        old_settings = dict(saved_identity.get("settings", {}))
        new_settings = dict(item["settings"])
        old_policy = old_settings.pop("post_compile_resource_policy", "reject")
        new_settings.pop("post_compile_resource_policy", None)
        if old_settings != new_settings or old_policy != "reject":
            raise ValueError(f"reuse attempt is not the audited reject-policy request: {output}")
        if not compatible_reuse_source_identity(saved_identity.get("source_identity", {}), code_identity):
            raise ValueError(f"reuse worker source is incompatible: {output}")

        monitor = read(output / "monitor.json")
        if (
            monitor.get("status") != "gpu_uncontended"
            or monitor.get("cpu_contention_policy") != "record"
            or gpu_hardware_signature(monitor.get("gpus", []))
            != gpu_hardware_signature(gpus[: item["gpu_count"]])
        ):
            raise ValueError(f"reuse attempt lacks compatible uncontended GPU telemetry: {output}")
        experiment = read(output / "experiment.json")
        if experiment.get("measurement", {}).get("backend") != item["settings"]["benchmark_backend"]:
            raise ValueError(f"reuse attempt used a different benchmark backend: {output}")
        summary = read(output / "summary.json")
        records = read(output / "outcomes.json")
        if summary.get("status") != "completed" or summary.get("config_count") != len(item["configs"]):
            raise ValueError(f"reuse summary is incomplete: {output}")
        if len(records) != len(item["configs"]):
            raise ValueError(f"reuse outcomes are incomplete: {output}")
        rerun_indices, reused_indices = [], []
        for index, (record, config) in enumerate(zip(records, item["configs"])):
            if (
                record.get("index") != index
                or record.get("config") != config
                or record.get("config_id") != config_id(config)
                or record.get("status") not in TERMINAL
            ):
                raise ValueError(f"invalid reuse outcome at index {index}: {output}")
            if record["status"] == "benchmarked" and (
                not isinstance(record.get("latency_ms"), int | float)
                or not math.isfinite(record["latency_ms"])
                or record["latency_ms"] <= 0
            ):
                raise ValueError(f"invalid reuse latency at index {index}: {output}")
            (rerun_indices if record["status"] == "post_compile_rejected" else reused_indices).append(index)
        specs[reuse_key(item)] = dict(
            schema_version=1,
            attempt=str(output.resolve()),
            request_id=completion["request_id"],
            completion_sha256=file_hash(marker),
            files=files,
            reused_indices=reused_indices,
            rerun_indices=rerun_indices,
            source_tuning_seconds=summary["tuning_seconds"],
            source_worker_wall_seconds=monitor.get("wall_seconds"),
            source_candidate_statuses=summary.get("candidate_statuses", {}),
        )
    return specs


def resolve_run_state(root, *, resume=False, resume_or_start=False):
    """Resolve strict CLI resume semantics, including an interrupted initialization."""
    root = Path(root)
    manifest = root / "manifest.json"
    if resume and resume_or_start:
        raise ValueError("--resume and --resume-or-start are mutually exclusive")
    if resume:
        if not manifest.is_file():
            raise ValueError("resume requires an existing frozen manifest")
        return True, False
    if resume_or_start and manifest.is_file():
        return True, False
    if root.exists():
        if not resume_or_start:
            raise FileExistsError("output already exists; use --resume to verify and continue it")
        if not root.is_dir():
            raise ValueError("output exists but is not a directory")
        entries = {path.name for path in root.iterdir()}
        if entries - {"manifest.json.tmp"}:
            raise ValueError("resume-or-start found an output directory without a frozen manifest")
        return False, True
    return False, False


def source_identity():
    from experiments.utils.cli import source_hashes
    from experiments.utils.baseline_store import measurement_sources
    from experiments.families import FAMILIES

    root = Path(__file__).resolve().parents[2]
    sources = {
        **source_hashes(
            "experiments/common/b200.py",
            "experiments/common/resource_policy.py",
            "experiments/common/run_b200_slurm.sh",
            "tilelang/tiletune/config.py",
            "tilelang/tiletune/runtime.py",
        ),
        **measurement_sources(set(FAMILIES.values())),
    }
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
                    settings=dict(
                        SETTINGS,
                        preflight=preflight,
                        post_compile_resource_policy=(
                            TILETUNE_RESOURCE_POLICY if experiment == "E3" else EXHAUSTIVE_RESOURCE_POLICY
                        ),
                    ),
                )
            )
    return rows


def select_gpus(observation, requested=None, *, required=4, gpu_model="B200"):
    from experiments.utils.monitor import visible_gpus

    if required not in (1, 4):
        raise ValueError("the study requires either one E1 GPU or four E2/E3 GPUs")
    if gpu_model not in ("B200", "B300"):
        raise ValueError("the study supports B200 or B300 GPUs")
    visible = visible_gpus(observation)
    if gpu_model == "B300":
        script = """
import json
import torch
identities = []
for device_index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(device_index)
    identities.append(dict(uuid="GPU-" + str(properties.uuid).removeprefix("GPU-"),
                           name=properties.name, compute_cap=f"{properties.major}.{properties.minor}"))
print(json.dumps(identities))
"""
        identities = json.loads(subprocess.check_output([sys.executable, "-c", script], text=True))
        by_uuid = {identity["uuid"]: identity for identity in identities}
        verified = []
        for gpu in visible:
            identity = by_uuid.get(gpu["uuid"])
            if identity and "B300" in identity["name"] and identity["compute_cap"] == gpu["compute_cap"] == "10.3":
                verified.append(dict(gpu, nvml_name=gpu["name"], name=identity["name"]))
        visible = verified
    if requested is not None:
        if len(requested) != required or len(set(requested)) != required:
            raise ValueError(f"this experiment selection requires exactly {required} distinct GPU index/indices")
        by_index = {int(g["index"]): g for g in visible}
        if set(requested) - by_index.keys():
            raise ValueError("requested GPUs are not all visible")
        selected = [by_index[i] for i in requested]
    else:
        selected = [gpu for gpu in visible if gpu_model in gpu["name"]][:required]
    if len(selected) != required or len({gpu["name"] for gpu in selected}) != 1 or any(gpu_model not in gpu["name"] for gpu in selected):
        raise ValueError(f"{required} matching {gpu_model} GPU(s) are required")
    return [{key: gpu[key] for key in ("index", "uuid", "name", "compute_cap", "nvml_name") if key in gpu} for gpu in selected]


def validate_attempt(output, request):
    """A worker summary alone never establishes a completed, uncontended sweep."""
    monitor = read(output / "monitor.json")
    if monitor.get("cpu_contention_policy") != "record" or monitor["status"] != "gpu_uncontended":
        raise ValueError("attempt has no GPU-uncontended, CPU-record-only monitor completion")
    if len(monitor.get("gpus", [])) != request["gpu_count"]:
        raise ValueError("attempt did not monitor exactly its active GPU subset")
    saved_request = read(output / "request.json")
    if any(saved_request.get(k) != v for k, v in request.items()):
        raise ValueError("attempt request identity mismatch")
    summary = read(output / "summary.json")
    if summary.get("status") != "completed" or summary.get("config_count") != len(request["configs"]):
        raise ValueError("incomplete worker summary")
    if summary.get("compiler_workers") != COMPILER_WORKERS or summary.get("benchmark_gpu_count") != request["gpu_count"]:
        raise ValueError("worker did not use the frozen CPU/GPU counts")
    experiment = read(output / "experiment.json")
    if request.get("source_identity") and experiment.get("source_identity") != request["source_identity"]:
        raise ValueError("worker source identity changed")
    if experiment.get("measurement", {}).get("backend") != request["settings"].get("benchmark_backend"):
        raise ValueError("worker did not use the frozen benchmark backend")
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
        if request["experiment"] in ("E1", "E2") and record["status"] == "post_compile_rejected":
            raise ValueError(f"{request['experiment']} cannot exclude a compiled candidate at the report-only resource check")
        if record["status"] == "benchmarked" and (
            not isinstance(record.get("latency_ms"), int | float) or not math.isfinite(record["latency_ms"]) or record["latency_ms"] <= 0
        ):
            raise ValueError("successful candidate requires a finite positive latency")
    winner = summary.get("winner_config")
    if not any(
        r["config"] == winner and r["status"] == "benchmarked" and r["latency_ms"] == summary.get("winner_latency_ms") for r in records
    ):
        raise ValueError("winner is absent from successful candidate measurements")
    reuse = request.get("reuse")
    if reuse:
        report = read(output / "reuse.json")
        if report.get("source") != reuse:
            raise ValueError("reuse provenance differs from the frozen request")
        rerun = reuse.get("rerun_indices", [])
        reused = reuse.get("reused_indices", [])
        if sorted(rerun + reused) != list(range(len(records))) or set(rerun) & set(reused):
            raise ValueError("reuse partition does not cover the candidate pool exactly once")
        if (
            report.get("rerun_indices") != rerun
            or report.get("reused_indices") != reused
            or summary.get("continuation_candidate_count") != len(rerun)
            or summary.get("reused_candidate_count") != len(reused)
        ):
            raise ValueError("reuse counts do not match the frozen candidate partition")
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
    output = case_dir / completion["attempt"]
    saved_request = read(output / "request.json")
    saved_identity = {key: value for key, value in saved_request.items() if key != "parent_pid"}
    if completion["request_id"] != digest(saved_identity) or not compatible_request(saved_identity, request):
        raise ValueError("completed workload belongs to a different request")
    for name, expected in completion["files"].items():
        if file_hash(output / name) != expected:
            raise ValueError(f"completed artifact changed: {output / name}")
    validate_attempt(output, saved_identity)
    return output


def run_queue(
    root,
    plan,
    gpus,
    cpu_ids,
    lease_fds,
    *,
    run=None,
    max_contention_retries=3,
    code_identity=None,
    reuse_specs=None,
):
    from experiments.utils.monitor import run_monitored

    run = run or run_monitored
    reuse_specs = reuse_specs or {}
    rows = []
    for item in plan:
        request = dict(item, cpu_ids=cpu_ids, source_identity=code_identity)
        if reuse := reuse_specs.get(reuse_key(item)):
            request["reuse"] = reuse
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
            attempt_metadata = dict(request_id=digest(request), slurm=slurm_metadata())
            write_json(output / "attempt.json", dict(status="running", **attempt_metadata))
            temporary = output / "tmp"
            temporary.mkdir()
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=",".join(g["uuid"] for g in active),
                TILELANG_DISABLE_CACHE="1",
                TILELANG_AUTO_TUNING_DISABLE_CACHE="1",
                TILELANG_AUTO_TUNING_CPU_COUNTS=str(COMPILER_WORKERS),
                TILELANG_AUTO_TUNING_MAX_CPU_COUNT=str(COMPILER_WORKERS),
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
                    # Compilation and benchmarking are separate. The frozen
                    # CPU set has eight logical CPUs beyond the 64 compiler
                    # workers; the coordinator/monitor runs outside this set.
                    # Shared-host CPU use is recorded but never invalidates
                    # CUDA-event results or restarts a workload.
                    max_external_busy_cores=CPU_HEADROOM,
                    cpu_contention_policy="record",
                )
                summary_path = output / "summary.json"
                if summary_path.is_file():
                    summary = read(summary_path)
                    current_wall = read(output / "monitor.json").get("wall_seconds")
                    summary["continuation_worker_wall_seconds"] = current_wall
                    summary["reused_worker_wall_seconds"] = (
                        request.get("reuse", {}).get("source_worker_wall_seconds", 0) or 0
                    )
                    summary["worker_wall_seconds"] = (
                        summary["reused_worker_wall_seconds"] + (current_wall or 0)
                    )
                    write_json(summary_path, summary)
                summary = validate_attempt(output, request)
            except BaseException as error:
                status = read(output / "monitor.json").get("status") if (output / "monitor.json").exists() else "failed"
                status = "interrupted" if isinstance(error, KeyboardInterrupt) else status
                if status not in RETRYABLE | {"interrupted", "timeout", "worker_failed"}:
                    status = "failed"
                write_json(output / "attempt.json", dict(status=status, error=str(error), **attempt_metadata))
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
                "attempt.json",
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
            else:
                files += ["resource-filter.tsv"]
                if request.get("reuse"):
                    files += ["reuse.json", "continuation-benchmarks.tsv", "continuation-resource-filter.tsv"]
            write_json(output / "attempt.json", dict(status="completed", **attempt_metadata))
            write_json(
                case_dir / "completed.json",
                dict(request_id=digest(request), attempt=output.name, files={name: file_hash(output / name) for name in files}),
            )
        summary = read(output / "summary.json")
        row = dict(
            summary,
            experiment=item["experiment"],
            attempt=str(output.relative_to(root)),
            worker_wall_seconds=summary.get(
                "worker_wall_seconds", read(output / "monitor.json").get("wall_seconds")
            ),
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


def _compiler_resource_observation(output, index):
    """Read exact post-compile counters for one exhaustive candidate."""
    rows = []
    with (Path(output) / "resource-filter.tsv").open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if int(row["index"]) == index and row["stage"] == "post_compile":
                details = json.loads(row["details"])
                rows.extend(details.get("kernels", []))
    fields = ("spill_stores_bytes", "spill_loads_bytes", "local_size_bytes", "n_regs")
    complete = bool(rows) and all(kernel.get(field) is not None for kernel in rows for field in fields)
    maxima = {
        field: max((kernel[field] for kernel in rows if kernel.get(field) is not None), default=None)
        for field in fields
    }
    classifications = sorted(
        {
            kernel.get("detected_kernel_type")
            for kernel in rows
            if kernel.get("detected_kernel_type") is not None
        }
    )
    return dict(complete=complete, kernels=len(rows), classifications=classifications, **maxima)


def oracle_resource_audit(root):
    """Block E3 unless every exact E1/E2 oracle fits its family policy."""
    root = Path(root)
    modes = ("E1", "E2")
    expected = {mode: study_plan(experiments=[mode]) for mode in modes}
    manifests = {}
    attempts = {}
    for mode in modes:
        mode_root = root / mode
        manifest_path = mode_root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"{mode} manifest is missing; E3 cannot establish oracle-safe limits")
        manifest = manifests[mode] = read(manifest_path)
        if manifest.get("plan") != expected[mode]:
            raise ValueError(f"{mode} manifest does not contain the current frozen {mode} plan")
        for item in expected[mode]:
            request = dict(
                item,
                cpu_ids=manifest["cpu_ids"],
                source_identity=manifest["source_identity"],
            )
            if reuse := manifest.get("reuse", {}).get(reuse_key(item)):
                request["reuse"] = reuse
            name = item["workload"]["name"]
            output = completed_attempt(mode_root / mode / name, request)
            if output is None:
                raise ValueError(f"{mode}/{name} is incomplete; E3 cannot establish oracle-safe limits")
            attempts[(mode, name)] = output
    for key in ("python", "revision"):
        if len({digest(manifests[mode].get(key)) for mode in modes}) != 1:
            raise ValueError(f"E1/E2 manifests disagree on {key}; E3 cannot use their joint oracle set")
    if len({digest(measurement_identity(manifests[mode].get("source_identity", {}))) for mode in modes}) != 1:
        raise ValueError("E1/E2 manifests disagree on measurement source_identity")

    from .resource_policy import b200_post_compile_policy

    rows = []
    for mode in modes:
        for item in expected[mode]:
            workload = Workload(**item["workload"])
            output = attempts[(mode, workload.name)]
            records = read(output / "outcomes.json")
            valid = [
                record
                for record in records
                if record["status"] == "benchmarked" and record.get("latency_ms", 0) > 0
            ]
            if not valid:
                raise ValueError(f"{mode}/{workload.name} has no measured oracle")
            best = min(record["latency_ms"] for record in valid)
            target = read(output / "experiment.json")["target"]
            policy = b200_post_compile_policy(workload, target)
            if policy is None:
                raise ValueError(f"{mode}/{workload.name} has no B200 policy for {target!r}")
            for oracle in (record for record in valid if record["latency_ms"] == best):
                observed = _compiler_resource_observation(output, oracle["index"])
                required_spill = max(
                    (observed["spill_stores_bytes"] or 0),
                    (observed["spill_loads_bytes"] or 0),
                )
                within_policy = (
                    observed["complete"]
                    and required_spill <= policy["max_spill_bytes"]
                    and observed["local_size_bytes"] <= policy["max_local_bytes"]
                )
                rows.append(
                    dict(
                        workload=workload.name,
                        family=FAMILIES[workload.op],
                        op=workload.op,
                        baseline=mode,
                        config_id=oracle["config_id"],
                        config=oracle["config"],
                        oracle_latency_ms=best,
                        observed=observed,
                        required_spill_bytes=required_spill,
                        max_spill_bytes=policy["max_spill_bytes"],
                        max_local_bytes=policy["max_local_bytes"],
                        within_policy=within_policy,
                    )
                )

    family_rows = {}
    for family in dict.fromkeys(row["family"] for row in rows):
        values = [row for row in rows if row["family"] == family]
        family_rows[family] = dict(
            oracle_records=len(values),
            max_observed_spill_bytes=max(row["required_spill_bytes"] for row in values),
            max_observed_local_bytes=max((row["observed"]["local_size_bytes"] or 0) for row in values),
            max_spill_bytes=values[0]["max_spill_bytes"],
            max_local_bytes=values[0]["max_local_bytes"],
            passed=all(row["within_policy"] for row in values),
        )
    audited_cases = {(row["baseline"], row["workload"]) for row in rows}
    required_cases = sum(len(expected[mode]) for mode in modes)
    report = dict(
        required_cases=required_cases,
        audited_cases=len(audited_cases),
        oracle_records=len(rows),
        complete=len(audited_cases) == required_cases,
        passed=len(audited_cases) == required_cases and all(row["within_policy"] for row in rows),
        families=family_rows,
        oracles=rows,
    )
    write_json(root / "oracle_resource_policy.json", report)
    if rows:
        with (root / "oracle_resource_policy.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return report


def oracle_audit(root, plan, *, mode_roots=None):
    """Retain every exact E1/E2 minimum and audit all four E3 retention stages."""
    if any(item["settings"]["preflight"] for item in plan) or {item["experiment"] for item in plan} != set(MODES):
        return None
    mode_roots = mode_roots or {mode: Path(root) for mode in MODES}
    by_name = {}
    for item in plan:
        path = Path(mode_roots[item["experiment"]]) / item["experiment"] / item["workload"]["name"]
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
                post_compile = record.get("post_compile") or {}
                resources = post_compile.get("resources") or {}
                resource_fields = ("spill_stores_bytes", "spill_loads_bytes", "local_bytes")
                resource_counters_complete = bool(resources) and all(
                    counters.get(field) is not None
                    for counters in resources.values()
                    for field in resource_fields
                )
                zero_spill_local = resource_counters_complete and all(
                    counters[field] == 0
                    for counters in resources.values()
                    for field in resource_fields
                )
                post_pass = post_compile.get("keep") is True and post_compile.get("status") == "pass"
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
                        post_compile_keep=post_compile.get("keep"),
                        post_compile_status=post_compile.get("status"),
                        post_compile_pass=post_pass,
                        post_compile_counters_complete=resource_counters_complete,
                        zero_spill_local_observed=zero_spill_local,
                        correctness_and_benchmark_pass=usable,
                        post_compile=post_compile,
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


def combine_study(root):
    """Validate three separately launched mode roots and build shared reports."""
    root = Path(root)
    mode_roots = {mode: root / mode for mode in MODES}
    expected = {mode: study_plan(experiments=[mode]) for mode in MODES}
    manifests = {mode: read(mode_roots[mode] / "manifest.json") for mode in MODES}
    for mode in MODES:
        if manifests[mode].get("plan") != expected[mode]:
            raise ValueError(f"{mode} manifest does not contain the current frozen {mode} plan")
    for key in ("python", "revision"):
        if len({digest(manifests[mode].get(key)) for mode in MODES}) != 1:
            raise ValueError(f"separate experiment manifests disagree on {key}")
    if len({digest(measurement_identity(manifests[mode].get("source_identity", {}))) for mode in MODES}) != 1:
        raise ValueError("separate experiment manifests disagree on measurement source_identity")
    resource_audit = oracle_resource_audit(root)
    if not resource_audit["passed"]:
        raise ValueError("one or more E1/E2 oracles exceed the frozen E3 compiler-resource policy")

    rows = []
    for mode in MODES:
        for item in expected[mode]:
            request = dict(
                item,
                cpu_ids=manifests[mode]["cpu_ids"],
                source_identity=manifests[mode]["source_identity"],
            )
            if reuse := manifests[mode].get("reuse", {}).get(reuse_key(item)):
                request["reuse"] = reuse
            output = completed_attempt(
                mode_roots[mode] / mode / item["workload"]["name"],
                request,
            )
            if output is None:
                raise ValueError(f"{mode}/{item['workload']['name']} is incomplete")
            summary = read(output / "summary.json")
            attempt_monitor = read(output / "monitor.json")
            rows.append(
                dict(
                    summary,
                    experiment=mode,
                    attempt=str(output.relative_to(root)),
                    worker_wall_seconds=summary.get("worker_wall_seconds", attempt_monitor.get("wall_seconds")),
                    gpu_uuids=[gpu["uuid"] for gpu in attempt_monitor["gpus"]],
                    slurm=read(output / "attempt.json").get("slurm", {}),
                )
            )

    baselines = {row["workload"]: row for row in rows if row["experiment"] == "E1"}
    for row in rows:
        baseline = baselines[row["workload"]]
        row["tuning_speedup_vs_E1"] = baseline["tuning_seconds"] / row["tuning_seconds"]
        row["end_to_end_speedup_vs_E1"] = baseline["worker_wall_seconds"] / row["worker_wall_seconds"]
    write_json(root / "comparison.json", rows)
    audit = oracle_audit(root, study_plan(), mode_roots=mode_roots)
    return dict(
        status="completed",
        workloads=len(baselines),
        rows=len(rows),
        resource_policy_audit={key: value for key, value in resource_audit.items() if key != "oracles"},
        oracle_audit={key: value for key, value in audit.items() if key != "oracles"},
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+")
    parser.add_argument("--gpu-model", choices=("B200", "B300"), default="B200")
    parser.add_argument("--output", type=Path)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", action="store_true")
    resume.add_argument(
        "--resume-or-start",
        action="store_true",
        help="resume a frozen output or initialize a missing/empty output; intended for restarted batch jobs",
    )
    parser.add_argument(
        "--combine",
        type=Path,
        metavar="STUDY_ROOT",
        help="combine completed E1, E2 and E3 subdirectories",
    )
    parser.add_argument(
        "--audit-resource-policy",
        type=Path,
        metavar="STUDY_ROOT",
        help="verify every exact E1/E2 oracle against the frozen per-family E3 limits",
    )
    parser.add_argument("--plan", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="one workload per selected family/mode, with at most eight compiled candidates per mode",
    )
    parser.add_argument("--workloads", nargs="+")
    parser.add_argument("--experiments", choices=list(MODES), nargs="+")
    parser.add_argument(
        "--reuse-from",
        type=Path,
        metavar="MODE_ROOT",
        help="reuse terminal outcomes from a validated E1/E2 mode root and rerun its post-compile rejects",
    )
    args = parser.parse_args(argv)
    if args.audit_resource_policy is not None:
        execution_options = (
            args.output,
            args.gpus,
            args.resume,
            args.resume_or_start,
            args.combine,
            args.plan,
            args.preflight,
            args.workloads,
            args.experiments,
            args.reuse_from,
        )
        if any(execution_options):
            parser.error("--audit-resource-policy cannot be used with other options")
        report = oracle_resource_audit(args.audit_resource_policy.resolve())
        print(json.dumps({key: value for key, value in report.items() if key != "oracles"}, indent=2))
        if not report["passed"]:
            raise ValueError("one or more E1/E2 oracles exceed the frozen E3 compiler-resource policy")
        return 0
    if args.combine is not None:
        execution_options = (
            args.output,
            args.gpus,
            args.resume,
            args.resume_or_start,
            args.plan,
            args.preflight,
            args.workloads,
            args.experiments,
            args.reuse_from,
            args.audit_resource_policy,
        )
        if any(execution_options):
            parser.error("--combine cannot be used with execution or planning options")
        print(json.dumps(combine_study(args.combine.resolve()), indent=2))
        return 0
    plan = study_plan(workloads=args.workloads, experiments=args.experiments, preflight=args.preflight)
    required_gpus = max(item["gpu_count"] for item in plan)
    if args.gpus is not None and (len(args.gpus) != required_gpus or len(set(args.gpus)) != required_gpus):
        parser.error(f"selected experiments require exactly {required_gpus} distinct GPU index/indices")
    if args.plan:
        print(json.dumps(dict(plan=plan, max_gpus=required_gpus, concurrent_workloads=1), indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required")
    from experiments.utils.isolation import measurement_lease, interruptible, select_cpu_ids
    from experiments.utils.monitor import snapshot

    root = args.output.resolve()
    resuming, initialize_existing = resolve_run_state(
        root,
        resume=args.resume,
        resume_or_start=args.resume_or_start,
    )
    frozen = read(root / "manifest.json") if resuming else None
    observed = snapshot()
    requested = args.gpus
    gpus = select_gpus(observed, requested, required=required_gpus, gpu_model=args.gpu_model)
    current_source_identity = source_identity()
    if args.reuse_from is not None and args.preflight:
        parser.error("--reuse-from cannot be used for a preflight-only run")
    if args.reuse_from is not None and any(item["experiment"] == "E3" for item in plan):
        parser.error("--reuse-from is supported only for E1/E2")
    reuse_specs = (
        prepare_reuse_specs(args.reuse_from, plan, current_source_identity, gpus)
        if args.reuse_from is not None
        else (frozen.get("reuse", {}) if frozen else {})
    )
    identity = dict(
        version=2,
        plan=plan,
        gpus=gpus,
        source_identity=current_source_identity,
        python=str(Path(sys.executable).resolve()),
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        reuse=reuse_specs,
    )
    if frozen:
        validate_resume_identity(frozen, identity)
    with measurement_lease(gpus) as leases, interruptible():
        allocation_cpus = os.sched_getaffinity(0)
        if frozen and set(frozen["cpu_ids"]) <= allocation_cpus and allocation_cpus - set(frozen["cpu_ids"]):
            cpu_ids = frozen["cpu_ids"]
        else:
            cpu_ids = select_cpu_ids(COMPILER_WORKERS)
            if frozen and len(cpu_ids) != len(frozen["cpu_ids"]):
                raise RuntimeError("resumed allocation cannot preserve the frozen CPU count")
        root.mkdir(parents=True, exist_ok=resuming or initialize_existing)
        # The coordinator and its monitor do not compete with compilation threads.
        affinity = allocation_cpus
        spare = affinity - set(cpu_ids)
        if not spare:
            raise RuntimeError("no CPU left for the measurement coordinator")
        # Publish a resumable identity only after the allocation can actually
        # satisfy the frozen worker/coordinator isolation contract.
        if not frozen:
            write_json(root / "manifest.json", dict(identity, cpu_ids=cpu_ids))
        os.sched_setaffinity(0, spare)
        try:
            if not args.preflight:
                preflight_root = root / "preflight"
                preflight_root.mkdir(exist_ok=True)
                preflight = study_plan(experiments=args.experiments, preflight=True)
                write_json(root / "progress.json", dict(status="preflight", total=len(plan), completed=0))
                run_queue(preflight_root, preflight, gpus, cpu_ids, leases, code_identity=identity["source_identity"])
            run_queue(
                root,
                plan,
                gpus,
                cpu_ids,
                leases,
                code_identity=identity["source_identity"],
                reuse_specs=reuse_specs,
            )
            audit = oracle_audit(root, plan)
            if audit is not None:
                print(json.dumps({k: v for k, v in audit.items() if k != "oracles"}), flush=True)
        finally:
            os.sched_setaffinity(0, affinity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
