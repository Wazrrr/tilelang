"""Run local CUDA workers with recorded, continuous contention checks."""

from contextlib import suppress
from datetime import datetime, timezone
import json
import os
import re
import signal
from pathlib import Path
import subprocess
import time

from experiments.utils.io import write_json


def snapshot():
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
            owned = worker_pid is not None and os.getpgid(int(row["pid"])) == worker_pid
        except ProcessLookupError:
            owned = False
        if not owned:
            foreign.append(row)
    return foreign


def exclusive_foreign_processes(observation, gpu_uuid, observed_pid):
    """Attribute ownership by exclusive launch, not by PID namespace.

    The coordinator starts a worker only after the GPU has been observed empty,
    so while the worker is alive a single stable NVML compute PID is the worker.
    NVML may report host PIDs that do not resolve inside a container PID
    namespace, so ownership must not depend on os.getpgid of the reported PID.
    """
    contexts = [row for row in observation["processes"] if row["gpu_uuid"] == gpu_uuid]
    if len(contexts) > 1:
        return contexts, None
    pid = contexts[0]["pid"] if contexts else None
    if pid is None or observed_pid is None or observed_pid == pid:
        return [], pid
    return contexts, None


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


def run_monitored(command, output, gpus, *, cwd=None, env=None, timeout=None, wait_idle=True):
    """Reject the whole invocation if a foreign process is observed on any GPU.

    Logs and partial artifacts remain inspectable; callers must not publish them
    as completed measurements. Process polling cannot detect subsecond overlap.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    process = None
    started = time.monotonic()
    observed_pids = {}
    audit = dict(
        gpus=gpus, poll_interval_seconds=1, status="waiting",
        ownership_policy="empty GPU before launch; one stable NVML compute PID per GPU during the worker",
        ownership_limitation="first context ownership is inferred from exclusive launch; polling cannot exclude subsecond interference",
    )
    try:
        with (output / "gpu_observations.jsonl").open("a") as observations, (output / "worker.log").open("w") as log:
            while True:
                observed = snapshot()
                foreign = []
                for gpu in gpus:
                    uuid = gpu["uuid"]
                    contexts = [p for p in observed["processes"] if p["gpu_uuid"] == uuid]
                    # NVML reports host PIDs, which may collide with unrelated
                    # local PIDs inside a container. Never use local /proc to
                    # attribute a context after an exclusive worker launch.
                    if process is None or len(contexts) > 1:
                        foreign.extend(contexts)
                    elif contexts:
                        pid = contexts[0]["pid"]
                        if uuid in observed_pids and observed_pids[uuid] != pid:
                            foreign.extend(contexts)
                        else:
                            observed_pids[uuid] = pid
                            audit["observed_nvml_pids"] = dict(observed_pids)
                observations.write(json.dumps(observed) + "\n")
                observations.flush()
                if process is None:
                    busy = any(float(next(g for g in observed["gpus"] if g["uuid"] == gpu["uuid"])["utilization.gpu"]) > 5 for gpu in gpus)
                    if foreign or busy:
                        if not wait_idle:
                            raise RuntimeError("requested GPU is busy")
                        time.sleep(1)
                        continue
                    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    started = time.monotonic()
                    audit.update(status="running", worker_pid=process.pid)
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
