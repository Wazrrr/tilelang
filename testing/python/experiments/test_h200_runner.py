"""Interruption, exclusion, contention and completion contracts without a GPU."""

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from experiments.common import h200
from experiments.utils import isolation, monitor
from experiments.utils.io import write_json


def devices(count=8):
    return [dict(index=str(i), uuid=f"GPU-{i}", name="NVIDIA H200", compute_cap="9.0", **{"utilization.gpu": "0"}) for i in range(count)]


def small_plan():
    plan = h200.study_plan(workloads=["gemm_decode", "gemm_prefill"], experiments=["E1"])
    for item in plan:
        item["indices"] = [0, 1]
        item["configs"] = item["configs"][:2]
    return plan


def successful_worker(command, output, gpus, **kwargs):
    request = h200.read(output / "request.json")
    assert len(gpus) == request["gpu_count"]
    assert kwargs["env"]["TILELANG_AUTO_TUNING_CPU_COUNTS"] == "128"
    assert kwargs["env"]["TILELANG_AUTO_TUNING_MAX_CPU_COUNT"] == "128"
    assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
    assert len(kwargs["env"]["CUDA_VISIBLE_DEVICES"].split(",")) <= 4
    write_json(output / "monitor.json", dict(status="uncontended", wall_seconds=2))
    write_json(output / "experiment.json", request)
    write_json(output / "compilation.json", {})
    (output / "benchmarks.tsv").write_text("index\tstatus\tlatency_ms\tconfig\terror\n")
    write_json(
        output / "outcomes.json",
        [
            dict(index=i, config=c, config_id=h200.config_id(c), status="benchmarked", latency_ms=i + 1)
            for i, c in enumerate(request["configs"])
        ],
    )
    write_json(
        output / "summary.json",
        dict(
            status="completed",
            config_count=len(request["configs"]),
            compiler_workers=128,
            benchmark_gpu_count=request["gpu_count"],
            winner_config=request["configs"][0],
            winner_latency_ms=1,
            workload=request["workload"]["name"],
            tuning_seconds=1,
        ),
    )


def test_plan_is_75_serial_workloads_with_identical_pools():
    plan = h200.study_plan()
    assert len(plan) == 75
    for index in range(25):
        e1, e2, e3 = [plan[index + offset] for offset in (0, 25, 50)]
        assert e1["configs"] == e2["configs"] == e3["configs"]
        assert [e["gpu_count"] for e in (e1, e2, e3)] == [1, 4, 4]
        assert e3["settings"]["group_size"] == 8
    assert sum(len(p["configs"]) for p in plan[:25]) == 28480
    preflight = h200.study_plan(preflight=True)
    assert len(preflight) == 15
    assert all(len(p["configs"]) == 8 for p in preflight if p["experiment"] != "E3")


def test_four_gpu_limit_respects_visibility_and_frozen_order(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    observation = dict(gpus=devices(), processes=[])
    assert [g["index"] for g in h200.select_gpus(observation)] == ["0", "1", "2", "3"]
    assert [g["index"] for g in h200.select_gpus(observation, [3, 2, 1, 0])] == ["3", "2", "1", "0"]
    for requested in ([0, 1, 2, 3, 4], [0, 1, 2], [0, 1, 1, 2], [0, 1, 2, 10]):
        with pytest.raises(ValueError):
            h200.select_gpus(observation, requested)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    assert [g["index"] for g in h200.select_gpus(observation)] == ["4", "5", "6", "7"]
    with pytest.raises(ValueError):
        h200.select_gpus(observation, [0, 1, 2, 3])


def test_interrupt_then_resume_preserves_completed_work_and_restarts_partial(tmp_path):
    plan, calls = small_plan(), []

    def interrupted(command, output, gpus, **kwargs):
        calls.append(str(output))
        if len(calls) == 2:
            write_json(output / "summary.json", dict(status="completed"))  # untrusted partial worker artifact
            raise KeyboardInterrupt("SIGTERM")
        successful_worker(command, output, gpus, **kwargs)

    with pytest.raises(KeyboardInterrupt):
        h200.run_queue(tmp_path, plan, devices(4), [0, 1], (), run=interrupted)
    first = tmp_path / "E1/gemm_decode/completed.json"
    before = first.read_bytes()
    assert not (tmp_path / "E1/gemm_prefill/completed.json").exists()
    assert h200.read(tmp_path / "progress.json")["status"] == "interrupted"
    rows = h200.run_queue(tmp_path, plan, devices(4), [0, 1], (), run=successful_worker)
    assert first.read_bytes() == before
    assert len(rows) == 2 and rows[1]["attempt"].endswith("attempt-0002")
    assert (tmp_path / "E1/gemm_prefill/attempt-0001/summary.json").is_file()
    assert h200.read(tmp_path / "progress.json")["status"] == "completed"


def test_resume_rejects_changed_pool_or_corrupted_completed_artifact(tmp_path):
    plan = small_plan()[:1]
    h200.run_queue(tmp_path, plan, devices(4), [0, 1], (), run=successful_worker)
    changed = deepcopy(plan)
    changed[0]["configs"].reverse()
    with pytest.raises(ValueError, match="different request"):
        h200.run_queue(tmp_path, changed, devices(4), [0, 1], (), run=successful_worker)
    (tmp_path / "E1/gemm_decode/attempt-0001/outcomes.json").write_text("[]")
    with pytest.raises(ValueError, match="artifact changed"):
        h200.run_queue(tmp_path, plan, devices(4), [0, 1], (), run=successful_worker)


def test_contention_retries_never_publish_rejected_timings(tmp_path):
    calls = []

    def contended(command, output, gpus, **kwargs):
        calls.append(output)
        successful_worker(command, output, gpus, **kwargs)
        if len(calls) == 1:
            write_json(output / "monitor.json", dict(status="host_contended"))
            raise RuntimeError("competing CPU job")

    rows = h200.run_queue(tmp_path, small_plan()[:1], devices(4), [0, 1], (), run=contended)
    assert len(calls) == 2
    assert rows[0]["attempt"].endswith("attempt-0002")
    assert h200.read(calls[0] / "attempt.json")["status"] == "host_contended"


def test_no_completion_marker_on_incomplete_outcomes(tmp_path):
    def incomplete(command, output, gpus, **kwargs):
        successful_worker(command, output, gpus, **kwargs)
        write_json(output / "outcomes.json", [])

    with pytest.raises(ValueError, match="incomplete candidate"):
        h200.run_queue(tmp_path, small_plan()[:1], devices(4), [0, 1], (), run=incomplete)
    assert not (tmp_path / "E1/gemm_decode/completed.json").exists()


def test_host_lease_excludes_second_launcher_even_with_disjoint_gpus(tmp_path, monkeypatch):
    monkeypatch.setattr(isolation.tempfile, "gettempdir", lambda: str(tmp_path))
    with (
        isolation.measurement_lease(devices(4)),
        pytest.raises(RuntimeError, match="no second workload"),
        isolation.measurement_lease(devices()[4:]),
    ):
        pytest.fail("second launcher entered")
    with isolation.measurement_lease(devices(4)):
        pass


def test_lease_survives_coordinator_close_until_child_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(isolation.tempfile, "gettempdir", lambda: str(tmp_path))
    with isolation.measurement_lease(devices(4)) as fds:
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], pass_fds=fds)
    try:
        with pytest.raises(RuntimeError), isolation.measurement_lease(devices(4)):
            pytest.fail("inherited lease was lost")
    finally:
        child.terminate()
        child.wait(timeout=5)
    with isolation.measurement_lease(devices(4)):
        pass


def test_cpu_monitor_excludes_own_compilation_and_counts_external_work(monkeypatch):
    observations = iter([{0: (1000, 100, 0, 0)}, {0: (1100, 180, 0, 0)}, {0: (1200, 260, 0, 0)}])
    owned = iter([0, 80, 100])
    timestamps = iter([0, 1, 2])
    monkeypatch.setattr(isolation, "cpu_ticks", lambda cpus: next(observations))
    monkeypatch.setattr(isolation, "owned_cpu_ticks", lambda pid: next(owned))
    monkeypatch.setattr(isolation.time, "monotonic", lambda: next(timestamps))
    monkeypatch.setattr(isolation.os, "sysconf", lambda name: 100)
    sampler = isolation.CpuMonitor([0])
    assert not sampler.sample()["ready"]
    assert sampler.sample(123)["external_busy_cores"] == 0
    assert sampler.sample(123)["external_busy_cores"] == pytest.approx(0.6)


def test_cpu_contention_kills_only_owned_worker(tmp_path, monkeypatch):
    gpu = devices(1)[0]
    samples = iter([0] * 5 + [3, 3])

    class Sampler:
        def __init__(self, cpus):
            pass

        def sample(self, pid):
            return dict(ready=True, external_busy_cores=next(samples), iowait_cores=0, steal_cores=0)

    class Process:
        pid = 999999
        killed = False

        def poll(self):
            return None

    process = Process()
    monkeypatch.setattr(isolation, "CpuMonitor", Sampler)
    monkeypatch.setattr(monitor, "snapshot", lambda: dict(gpus=[gpu], processes=[]))
    monkeypatch.setattr(monitor.time, "sleep", lambda _: None)
    monkeypatch.setattr(monitor.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(monitor, "stop_worker", lambda p: setattr(p, "killed", True))
    with pytest.raises(RuntimeError, match="discard this invocation"):
        monitor.run_monitored(["worker"], tmp_path, [gpu], cpu_ids=[0])
    assert process.killed
    assert h200.read(tmp_path / "monitor.json")["status"] == "host_contended"


def test_sigterm_worker_cleans_its_compiler_process_group(tmp_path):
    script = """
import os, subprocess, sys, time
from pathlib import Path
from experiments.utils.isolation import prepare_worker
prepare_worker(list(os.sched_getaffinity(0)), int(sys.argv[1]))
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
Path(sys.argv[2]).write_text(str(child.pid))
time.sleep(30)
"""
    marker = tmp_path / "child.pid"
    worker = subprocess.Popen([sys.executable, "-c", script, str(os.getpid()), str(marker)], start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists()
        worker.terminate()
        worker.wait(timeout=5)
        child = Path(f"/proc/{marker.read_text()}/stat")

        def alive():
            try:
                return child.read_text().rsplit(")", 1)[1].split()[0] != "Z"
            except (FileNotFoundError, ProcessLookupError):
                return False

        deadline = time.monotonic() + 5
        while alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not alive()
    finally:
        monitor.stop_worker(worker)


@pytest.mark.parametrize("failure", [None, "rank", "selection", "post_compile", "benchmark"])
def test_oracle_audit_checks_union_ties_and_every_retention_stage(tmp_path, failure):
    plan = h200.study_plan(workloads=["gemm_decode"])
    configs = plan[0]["configs"][:8]
    for item in plan:
        folder = tmp_path / item["experiment"] / "gemm_decode"
        output = folder / "attempt-0001"
        output.mkdir(parents=True)
        write_json(folder / "completed.json", dict(attempt=output.name))
        records = [
            dict(index=i, config=c, config_id=h200.config_id(c), status="benchmarked", latency_ms=10, post_compile=dict(status="pass"))
            for i, c in enumerate(configs)
        ]
        if item["experiment"] == "E1":
            records[0]["latency_ms"] = records[1]["latency_ms"] = 1
        elif item["experiment"] == "E2":
            records[2]["latency_ms"] = 1
        else:
            ranking = [dict(index=i, score=i + 1, tier="eligible", tie_last_rank=i + 1) for i in range(8)]
            selected = [0, 1, 2, 3]
            if failure == "rank":
                ranking[1]["tie_last_rank"] = 5
            elif failure == "selection":
                selected.remove(1)
            elif failure == "post_compile":
                records[1]["post_compile"]["status"] = "unknown"
            elif failure == "benchmark":
                records[1]["status"] = "benchmark_error"
            write_json(output / "tiletune.json", dict(ranking=ranking, selection=dict(selected_indices=selected)))
        write_json(output / "outcomes.json", records)
    report = h200.oracle_audit(tmp_path, plan)
    assert report["complete"] and report["workloads_audited"] == 1
    assert len(report["oracles"]) == 3
    assert report["workloads_retained"] == (1 if failure is None else 0)
    assert (tmp_path / "oracle_retention.csv").is_file()
