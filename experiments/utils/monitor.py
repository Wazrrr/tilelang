"""Run local CUDA workers with recorded, continuous contention checks."""

from contextlib import contextmanager, suppress
from datetime import datetime, timezone
import json
import os
import re
import signal
from pathlib import Path
import subprocess
import time

from experiments.utils.io import write_json


MAX_POLL_GAP_SECONDS = 5
MAX_HOST_LOAD_PER_CPU = 2


def snapshot():
    """Observe occupancy without starting a CUDA context or a subprocess per poll."""
    from .nvml import snapshot as nvml_snapshot

    started = time.monotonic()
    try:
        observation = nvml_snapshot()
    except (OSError, AttributeError):
        observation = _snapshot_smi()
    observation["duration_seconds"] = time.monotonic() - started
    observation["host_load_1m"] = os.getloadavg()[0]
    observation["host_cpu_count"] = len(os.sched_getaffinity(0))
    return observation


def host_overloaded(observation):
    return observation.get("host_load_1m", 0) > MAX_HOST_LOAD_PER_CPU * observation.get("host_cpu_count", os.cpu_count() or 1)


def poll_gap(previous, started, finished):
    return max(finished - started, finished - previous if previous is not None else 0)


def _snapshot_smi():
    def query(flag, fields):
        output = subprocess.check_output(["nvidia-smi", f"--{flag}={','.join(fields)}", "--format=csv,noheader,nounits"], text=True)
        return [dict(zip(fields, (v.strip() for v in row.split(",")))) for row in output.splitlines() if row.strip()]

    return dict(
        timestamp=datetime.now(timezone.utc).isoformat(),
        gpus=query(
            "query-gpu",
            ["index", "uuid", "name", "compute_cap", "utilization.gpu", "memory.used", "clocks.sm", "temperature.gpu", "power.draw"],
        ),
        processes=query("query-compute-apps", ["gpu_uuid", "pid", "process_name", "used_memory"]),
    )


def foreign_processes(observation, gpu_uuid, worker_pid=None):
    foreign = []
    for row in observation["processes"]:
        if row["gpu_uuid"] != gpu_uuid:
            continue
        try:
            owned = worker_pid is not None and (int(row["pid"]) == worker_pid or os.getpgid(int(row["pid"])) == worker_pid)
        except ProcessLookupError:
            owned = False  # A process observed on the GPU cannot be ignored just because it exited.
        if not owned:
            foreign.append(row)
    return foreign


def stop_worker(process):
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def visible_gpus(observation):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return observation["gpus"]
    result = []
    for token in visible.split(","):
        matches = [g for g in observation["gpus"] if token == g["index"] or g["uuid"].startswith(token) and token]
        if len(matches) != 1:
            raise ValueError(f"CUDA_VISIBLE_DEVICES entry cannot be resolved: {token!r}")
        result.append(matches[0])
    return result


def idle_gpus(observation, *, ignore_pid=None):
    if host_overloaded(observation) or observation.get("duration_seconds", 0) > MAX_POLL_GAP_SECONDS:
        return []
    return [
        g
        for g in visible_gpus(observation)
        if not [p for p in foreign_processes(observation, g["uuid"]) if int(p["pid"]) != ignore_pid] and float(g["utilization.gpu"]) <= 5
    ]


def matches_cuda_device(gpu, device):
    arch = device.target["arch"].removeprefix("sm_").rstrip("af")
    return gpu["compute_cap"].replace(".", "") == arch and bool(
        re.search(device.expected_device_pattern or ".*", gpu["name"], re.IGNORECASE)
    )


def select_cuda_gpu(device):
    """Select within the caller's visible set using architecture and model."""
    # A coordinator may retain an idle context after primitive profiling.
    # The monitor also excludes that PID while rejecting every foreign worker.
    for gpu in idle_gpus(snapshot(), ignore_pid=os.getpid()):
        if matches_cuda_device(gpu, device):
            return gpu
    raise RuntimeError(f"no matching idle visible CUDA GPU for {device.name} ({device.target['arch']})")


@contextmanager
def cuda_device(device):
    """Bind one study target, restoring visibility before the next target."""
    if device.worker or device.target["kind"] != "cuda":
        yield
        return
    gpu = select_cuda_gpu(device)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    try:
        yield
    finally:
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible


def run_monitored(command, output, gpus, *, cwd=None, env=None, timeout=None, wait_idle=True):
    """Reject the whole invocation if a foreign process is observed on any GPU.

    Logs and partial artifacts remain inspectable; callers must not publish them
    as completed measurements. Process polling cannot detect subsecond overlap.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    process = None
    started = time.monotonic()
    previous_poll = None
    audit = dict(gpus=gpus, poll_interval_seconds=1, max_poll_gap_seconds=MAX_POLL_GAP_SECONDS, status="waiting")
    try:
        with (output / "gpu_observations.jsonl").open("a") as observations, (output / "worker.log").open("w") as log:
            while True:
                poll_started = time.monotonic()
                observed = snapshot()
                poll_finished = time.monotonic()
                gap = poll_gap(previous_poll, poll_started, poll_finished)
                previous_poll = poll_finished
                delayed = gap > MAX_POLL_GAP_SECONDS
                overloaded = host_overloaded(observed)
                observed["poll_gap_seconds"] = gap
                foreign = [
                    p
                    for gpu in gpus
                    for p in foreign_processes(observed, gpu["uuid"], process.pid if process else None)
                    if int(p["pid"]) != os.getpid()
                ]
                observations.write(json.dumps(observed) + "\n")
                observations.flush()
                if process is None:
                    busy = any(float(next(g for g in observed["gpus"] if g["uuid"] == gpu["uuid"])["utilization.gpu"]) > 5 for gpu in gpus)
                    if foreign or busy or delayed or overloaded:
                        if not wait_idle:
                            raise RuntimeError("requested GPU is busy")
                        time.sleep(1)
                        continue
                    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    started = time.monotonic()
                    audit.update(status="running", worker_pid=process.pid)
                elif delayed or overloaded:
                    audit.update(
                        status="monitor_gap" if delayed else "host_contended",
                        observed_poll_gap_seconds=gap,
                        host_load_1m=observed.get("host_load_1m"),
                    )
                    raise RuntimeError("monitoring gap or host overload; discard this invocation's measurements")
                elif foreign:
                    audit.update(status="contended", foreign_processes=foreign)
                    raise RuntimeError("foreign GPU process observed; discard this invocation's measurements")
                elif process.poll() is not None:
                    audit.update(status="uncontended" if process.returncode == 0 else "worker_failed", exit_code=process.returncode)
                    if process.returncode:
                        raise RuntimeError(f"worker exited with code {process.returncode}; see {output / 'worker.log'}")
                    return
                if timeout is not None and time.monotonic() - started > timeout:
                    audit["status"] = "timeout"
                    raise TimeoutError(f"worker exceeded {timeout} seconds")
                time.sleep(1)
    finally:
        if process is not None and process.poll() is None:
            stop_worker(process)
        audit["wall_seconds"] = time.monotonic() - started
        write_json(output / "monitor.json", audit)
