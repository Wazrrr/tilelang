"""Interruption, exclusion, contention and completion contracts without a GPU."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from experiments.common import b200
from experiments.utils import isolation, monitor
from experiments.utils.io import write_json


def devices(count=8, *, start=0):
    return [
        dict(index=str(i - start), uuid=f"GPU-{i}", name="NVIDIA B200", compute_cap="10.0", **{"utilization.gpu": "0"})
        for i in range(start, start + count)
    ]


def small_plan():
    plan = b200.study_plan(workloads=["gemm_decode", "gemm_prefill"], experiments=["E1"])
    for item in plan:
        item["indices"] = [0, 1]
        item["configs"] = item["configs"][:2]
    return plan


def test_worker_import_guard_binds_tilelang_and_core_to_this_worktree():
    root = Path(__file__).resolve().parents[3]
    script = """
import json
from pathlib import Path
from experiments.utils.imports import use_local_tilelang
use_local_tilelang()
import tilelang, tiletune_core
print(json.dumps([str(Path(tilelang.__file__).resolve()), str(Path(tiletune_core.__file__).resolve())]))
"""
    paths = json.loads(subprocess.check_output([sys.executable, "-c", script], cwd=root, text=True).splitlines()[-1])
    assert paths == [str(root / "tilelang/__init__.py"), str(root / "tiletune_core/__init__.py")]


def successful_worker(command, output, gpus, **kwargs):
    request = b200.read(output / "request.json")
    assert len(gpus) == request["gpu_count"]
    assert kwargs["env"]["TILELANG_AUTO_TUNING_CPU_COUNTS"] == "64"
    assert kwargs["env"]["TILELANG_AUTO_TUNING_MAX_CPU_COUNT"] == "64"
    assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
    assert kwargs["max_external_busy_cores"] == 8
    assert kwargs["cpu_contention_policy"] == "record"
    assert len(kwargs["env"]["CUDA_VISIBLE_DEVICES"].split(",")) <= 4
    write_json(
        output / "monitor.json",
        dict(status="gpu_uncontended", cpu_contention_policy="record", wall_seconds=2, gpus=gpus),
    )
    write_json(
        output / "experiment.json",
        dict(request, measurement=dict(backend=request["settings"]["benchmark_backend"])),
    )
    write_json(output / "compilation.json", {})
    (output / "benchmarks.tsv").write_text("index\tstatus\tlatency_ms\tconfig\terror\n")
    if request["experiment"] != "E3":
        (output / "resource-filter.tsv").write_text("index\tstage\tverdict\treason\tconfig\tdetails\n")
    write_json(
        output / "outcomes.json",
        [
            dict(index=i, config=c, config_id=b200.config_id(c), status="benchmarked", latency_ms=i + 1)
            for i, c in enumerate(request["configs"])
        ],
    )
    write_json(
        output / "summary.json",
        dict(
            status="completed",
            config_count=len(request["configs"]),
            compiler_workers=64,
            benchmark_gpu_count=request["gpu_count"],
            winner_config=request["configs"][0],
            winner_latency_ms=1,
            workload=request["workload"]["name"],
            tuning_seconds=1,
        ),
    )


def test_plan_is_75_serial_workloads_with_identical_pools():
    plan = b200.study_plan()
    assert len(plan) == 75
    for index in range(25):
        e1, e2, e3 = [plan[index + offset] for offset in (0, 25, 50)]
        assert e1["configs"] == e2["configs"] == e3["configs"]
        assert [e["gpu_count"] for e in (e1, e2, e3)] == [1, 4, 4]
        assert e3["settings"]["group_size"] == 8
        assert e1["settings"]["post_compile_resource_policy"] == "report"
        assert e2["settings"]["post_compile_resource_policy"] == "report"
        assert e3["settings"]["post_compile_resource_policy"] == "reject"
        assert all(
            experiment["settings"]["benchmark_backend"] == "event"
            for experiment in (e1, e2, e3)
        )
    assert sum(len(p["configs"]) for p in plan[:25]) == 18075
    preflight = b200.study_plan(preflight=True)
    assert len(preflight) == 15
    assert all(len(p["configs"]) == 8 for p in preflight if p["experiment"] != "E3")


def test_resume_or_start_distinguishes_new_resumable_and_partial_roots(tmp_path):
    output = tmp_path / "run"
    assert b200.resolve_run_state(output, resume_or_start=True) == (False, False)
    output.mkdir()
    assert b200.resolve_run_state(output, resume_or_start=True) == (False, True)
    (output / "manifest.json.tmp").write_text("partial")
    assert b200.resolve_run_state(output, resume_or_start=True) == (False, True)
    write_json(output / "manifest.json", {})
    assert b200.resolve_run_state(output, resume_or_start=True) == (True, False)
    with pytest.raises(ValueError, match="mutually exclusive"):
        b200.resolve_run_state(output, resume=True, resume_or_start=True)

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "attempt").mkdir()
    with pytest.raises(ValueError, match="without a frozen manifest"):
        b200.resolve_run_state(broken, resume_or_start=True)


def test_resume_accepts_new_gpu_uuids_but_not_different_hardware_or_measurement_code():
    old_source = dict(
        sources={"experiments/common/b200.py": "old-coordinator", "experiments/common/system.py": "worker"},
        native_build={"build/lib/libtilelang.so": {"size": 1, "mtime_ns": 2}},
    )
    new_source = deepcopy(old_source)
    new_source["sources"]["experiments/common/b200.py"] = "new-coordinator"
    frozen = dict(version=1, plan=[{"workload": "test"}], gpus=devices(4), source_identity=old_source, python="/python", revision="rev")
    current = dict(frozen, gpus=devices(4, start=4), source_identity=new_source)
    b200.validate_resume_identity(frozen, current)

    changed_model = deepcopy(current)
    changed_model["gpus"][0]["name"] = "NVIDIA H200"
    with pytest.raises(ValueError, match="GPU count, model or compute capability"):
        b200.validate_resume_identity(frozen, changed_model)

    changed_code = deepcopy(current)
    changed_code["source_identity"]["sources"]["experiments/common/system.py"] = "changed-worker"
    with pytest.raises(ValueError, match="measurement code"):
        b200.validate_resume_identity(frozen, changed_code)


def test_completed_work_survives_coordinator_only_source_change(tmp_path):
    plan = small_plan()[:1]
    old_source = dict(sources={"experiments/common/b200.py": "old", "worker.py": "same"}, native_build={})
    new_source = dict(sources={"experiments/common/b200.py": "new", "worker.py": "same"}, native_build={})
    b200.run_queue(tmp_path, plan, devices(4), [0, 1], (), run=successful_worker, code_identity=old_source)

    def must_not_run(*args, **kwargs):
        pytest.fail("completed workload was rerun after an orchestration-only change")

    rows = b200.run_queue(tmp_path, plan, devices(4, start=4), [2, 3], (), run=must_not_run, code_identity=new_source)
    assert len(rows) == 1 and rows[0]["attempt"].endswith("attempt-0001")

    with pytest.raises(ValueError, match="different request"):
        b200.run_queue(tmp_path, plan, devices(4, start=4), [2, 3, 4], (), run=must_not_run, code_identity=new_source)


def test_combine_study_validates_three_separate_mode_roots(tmp_path, monkeypatch):
    plans = {}
    for mode in b200.MODES:
        item = deepcopy(b200.study_plan(workloads=["gemm_decode"], experiments=[mode])[0])
        item["indices"] = item["indices"][:2]
        item["configs"] = item["configs"][:2]
        plans[mode] = [item]

    def fake_plan(*, workloads=None, experiments=None, preflight=False):
        assert workloads is None and not preflight
        modes = list(experiments or b200.MODES)
        return [deepcopy(item) for mode in modes for item in plans[mode]]

    monkeypatch.setattr(b200, "study_plan", fake_plan)
    source_identity = dict(sources={"test": "hash"}, native_build={})
    for mode, gpu_count in (("E1", 1), ("E2", 4), ("E3", 4)):
        mode_root = tmp_path / mode
        mode_root.mkdir()
        item = plans[mode][0]
        cpu_ids = list(range(72))
        gpus = devices(gpu_count)
        attempt_gpus = devices(gpu_count, start=4)
        write_json(
            mode_root / "manifest.json",
            dict(
                plan=plans[mode],
                cpu_ids=cpu_ids,
                gpus=gpus,
                source_identity=source_identity,
                python="/python",
                revision="revision",
            ),
        )
        case = mode_root / mode / "gemm_decode"
        output = case / "attempt-0001"
        output.mkdir(parents=True)
        request = dict(item, cpu_ids=cpu_ids, source_identity=source_identity)
        write_json(output / "request.json", request)
        write_json(
            output / "attempt.json",
            dict(status="completed", request_id=b200.digest(request), slurm={}),
        )
        write_json(
            output / "monitor.json",
            dict(status="gpu_uncontended", cpu_contention_policy="record", wall_seconds=2, gpus=attempt_gpus),
        )
        write_json(
            output / "experiment.json",
            dict(
                source_identity=source_identity,
                measurement=dict(backend="event"),
                target=dict(kind="cuda", arch="sm_100a"),
            ),
        )
        records = [
            dict(
                index=index,
                config=config,
                config_id=b200.config_id(config),
                status="benchmarked" if mode != "E3" or index == 0 else "not_selected",
                latency_ms=index + 1 if mode != "E3" or index == 0 else None,
                post_compile=dict(
                    keep=True,
                    status="pass",
                    resources={
                        "kernel": dict(
                            spill_stores_bytes=0,
                            spill_loads_bytes=0,
                            local_bytes=0,
                        )
                    },
                ),
            )
            for index, config in enumerate(item["configs"])
        ]
        write_json(output / "outcomes.json", records)
        if mode != "E3":
            details = json.dumps(
                {
                    "kernels": [
                        dict(
                            detected_kernel_type="gemm",
                            spill_stores_bytes=0,
                            spill_loads_bytes=0,
                            local_size_bytes=0,
                            n_regs=168,
                        )
                    ]
                }
            )
            (output / "resource-filter.tsv").write_text(
                "index\tstage\tverdict\treason\tconfig\tdetails\n"
                + "".join(
                    f"{index}\tpost_compile\tkeep\tpassed\t{json.dumps(config)}\t{details}\n"
                    for index, config in enumerate(item["configs"])
                )
            )
        write_json(
            output / "summary.json",
            dict(
                status="completed",
                config_count=2,
                compiler_workers=64,
                benchmark_gpu_count=gpu_count,
                winner_config=item["configs"][0],
                winner_latency_ms=1,
                workload="gemm_decode",
                tuning_seconds=1,
            ),
        )
        if mode == "E3":
            selection = dict(alpha=0.5, strict_budget=True, pool_size=2, selected_indices=[0])
            write_json(output / "selection.json", selection)
            write_json(
                output / "tiletune.json",
                dict(
                    selection=selection,
                    ranking=[
                        dict(index=0, score=1, tier="eligible", tie_last_rank=1),
                        dict(index=1, score=2, tier="eligible", tie_last_rank=2),
                    ],
                ),
            )
        write_json(case / "completed.json", dict(request_id=b200.digest(request), attempt=output.name, files={}))

    result = b200.combine_study(tmp_path)
    assert result["status"] == "completed" and result["rows"] == 3
    assert b200.read(tmp_path / "comparison.json")[2]["gpu_uuids"] == [f"GPU-{i}" for i in range(4, 8)]
    assert b200.read(tmp_path / "oracle_retention.json")["complete"]
    resource_report = tmp_path / "E2/E2/gemm_decode/attempt-0001/resource-filter.tsv"
    resource_report.write_text(resource_report.read_text().replace('"spill_stores_bytes": 0', '"spill_stores_bytes": 1'))
    failed = b200.oracle_resource_audit(tmp_path)
    assert not failed["passed"]
    assert failed["families"]["gemm"]["max_observed_spill_bytes"] == 1


def test_four_gpu_limit_respects_visibility_and_frozen_order(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    observation = dict(gpus=devices(), processes=[])
    assert [g["index"] for g in b200.select_gpus(observation)] == ["0", "1", "2", "3"]
    assert [g["index"] for g in b200.select_gpus(observation, [3, 2, 1, 0])] == ["3", "2", "1", "0"]
    assert [g["index"] for g in b200.select_gpus(observation, [2], required=1)] == ["2"]
    for requested in ([0, 1, 2, 3, 4], [0, 1, 2], [0, 1, 1, 2], [0, 1, 2, 10]):
        with pytest.raises(ValueError):
            b200.select_gpus(observation, requested)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    assert [g["index"] for g in b200.select_gpus(observation)] == ["4", "5", "6", "7"]
    with pytest.raises(ValueError):
        b200.select_gpus(observation, [0, 1, 2, 3])


def test_interrupt_then_resume_preserves_completed_work_and_restarts_partial(tmp_path):
    plan, calls = small_plan(), []

    def interrupted(command, output, gpus, **kwargs):
        calls.append(str(output))
        if len(calls) == 2:
            write_json(output / "summary.json", dict(status="completed"))  # untrusted partial worker artifact
            raise KeyboardInterrupt("SIGTERM")
        successful_worker(command, output, gpus, **kwargs)

    with pytest.raises(KeyboardInterrupt):
        b200.run_queue(tmp_path, plan, devices(4), [0, 1], (), run=interrupted)
    first = tmp_path / "E1/gemm_decode/completed.json"
    before = first.read_bytes()
    assert not (tmp_path / "E1/gemm_prefill/completed.json").exists()
    assert b200.read(tmp_path / "progress.json")["status"] == "interrupted"
    rows = b200.run_queue(tmp_path, plan, devices(4), [0, 1], (), run=successful_worker)
    assert first.read_bytes() == before
    assert len(rows) == 2 and rows[1]["attempt"].endswith("attempt-0002")
    assert (tmp_path / "E1/gemm_prefill/attempt-0001/summary.json").is_file()
    assert b200.read(tmp_path / "progress.json")["status"] == "completed"


def test_resume_rejects_changed_pool_or_corrupted_completed_artifact(tmp_path):
    plan = small_plan()[:1]
    b200.run_queue(tmp_path, plan, devices(4), [0, 1], (), run=successful_worker)
    changed = deepcopy(plan)
    changed[0]["configs"].reverse()
    with pytest.raises(ValueError, match="different request"):
        b200.run_queue(tmp_path, changed, devices(4), [0, 1], (), run=successful_worker)
    (tmp_path / "E1/gemm_decode/attempt-0001/outcomes.json").write_text("[]")
    with pytest.raises(ValueError, match="artifact changed"):
        b200.run_queue(tmp_path, plan, devices(4), [0, 1], (), run=successful_worker)


def test_contention_retries_never_publish_rejected_timings(tmp_path):
    calls = []

    def contended(command, output, gpus, **kwargs):
        calls.append(output)
        successful_worker(command, output, gpus, **kwargs)
        if len(calls) == 1:
            write_json(output / "monitor.json", dict(status="host_contended"))
            raise RuntimeError("competing CPU job")

    rows = b200.run_queue(tmp_path, small_plan()[:1], devices(4), [0, 1], (), run=contended)
    assert len(calls) == 2
    assert rows[0]["attempt"].endswith("attempt-0002")
    assert b200.read(calls[0] / "attempt.json")["status"] == "host_contended"


def test_no_completion_marker_on_incomplete_outcomes(tmp_path):
    def incomplete(command, output, gpus, **kwargs):
        successful_worker(command, output, gpus, **kwargs)
        write_json(output / "outcomes.json", [])

    with pytest.raises(ValueError, match="incomplete candidate"):
        b200.run_queue(tmp_path, small_plan()[:1], devices(4), [0, 1], (), run=incomplete)
    assert not (tmp_path / "E1/gemm_decode/completed.json").exists()


@pytest.mark.parametrize("experiment,gpu_count", [("E1", 1), ("E2", 4)])
def test_exhaustive_completion_rejects_post_compile_exclusion(tmp_path, experiment, gpu_count):
    def incorrectly_pruned(command, output, gpus, **kwargs):
        successful_worker(command, output, gpus, **kwargs)
        records = b200.read(output / "outcomes.json")
        records[1].update(status="post_compile_rejected", latency_ms=None)
        write_json(output / "outcomes.json", records)

    plan = b200.study_plan(workloads=["gemm_decode"], experiments=[experiment])
    plan[0]["indices"] = [0, 1]
    plan[0]["configs"] = plan[0]["configs"][:2]
    with pytest.raises(ValueError, match="cannot exclude a compiled candidate at the report-only resource check"):
        b200.run_queue(tmp_path, plan, devices(gpu_count), [0, 1], (), run=incorrectly_pruned)
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


def test_machine_load_is_not_scaled_by_coordinator_affinity(monkeypatch):
    from experiments.utils import nvml

    monkeypatch.setattr(nvml, "snapshot", lambda: dict(gpus=[], processes=[]))
    monkeypatch.setattr(monitor.os, "getloadavg", lambda: (18, 0, 0))
    monkeypatch.setattr(monitor.os, "cpu_count", lambda: 224)
    monkeypatch.setattr(monitor.os, "sched_getaffinity", lambda pid: pytest.fail("narrow coordinator affinity used for host load"))
    observation = monitor.snapshot()
    assert observation["host_cpu_count"] == 224
    assert not monitor.host_overloaded(observation)


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
    assert b200.read(tmp_path / "monitor.json")["status"] == "host_contended"


def test_cpu_headroom_limit_is_recorded_and_enforced(tmp_path, monkeypatch):
    gpu = devices(1)[0]
    samples = iter([0] * 5 + [8, 8, 9, 9])

    class Sampler:
        def __init__(self, cpus):
            pass

        def sample(self, pid):
            return dict(ready=True, external_busy_cores=next(samples), iowait_cores=0, steal_cores=0)

    class Process:
        pid = 999999

        def poll(self):
            return None

    monkeypatch.setattr(isolation, "CpuMonitor", Sampler)
    monkeypatch.setattr(monitor, "snapshot", lambda: dict(gpus=[gpu], processes=[]))
    monkeypatch.setattr(monitor.time, "sleep", lambda _: None)
    monkeypatch.setattr(monitor.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(monitor, "stop_worker", lambda process: None)
    with pytest.raises(RuntimeError, match="discard this invocation"):
        monitor.run_monitored(["worker"], tmp_path, [gpu], cpu_ids=[0], max_external_busy_cores=8)
    audit = b200.read(tmp_path / "monitor.json")
    assert audit["max_external_busy_cores"] == 8
    assert audit["status"] == "host_contended"


def test_record_only_cpu_contention_does_not_reject_gpu_clean_work(tmp_path, monkeypatch):
    gpu = devices(1)[0]

    class Sampler:
        def __init__(self, cpus):
            pass

        def sample(self, pid):
            return dict(ready=True, external_busy_cores=12, iowait_cores=2, steal_cores=0.2)

    class Process:
        pid = 999999
        returncode = 0

        def poll(self):
            return 0

        def wait(self):
            return 0

    monkeypatch.setattr(isolation, "CpuMonitor", Sampler)
    monkeypatch.setattr(monitor, "snapshot", lambda: dict(gpus=[gpu], processes=[], host_load_1m=1000, host_cpu_count=1))
    monkeypatch.setattr(monitor.time, "sleep", lambda _: None)
    monkeypatch.setattr(monitor.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(monitor, "stop_worker", lambda process: None)
    monitor.run_monitored(["worker"], tmp_path, [gpu], cpu_ids=[0], cpu_contention_policy="record")
    audit = b200.read(tmp_path / "monitor.json")
    assert audit["status"] == "gpu_uncontended"
    assert audit["cpu_contention_observed"]
    assert audit["cpu_telemetry"]["max_external_busy_cores"] == 12


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
    plan = b200.study_plan(workloads=["gemm_decode"])
    configs = plan[0]["configs"][:8]
    for item in plan:
        folder = tmp_path / item["experiment"] / "gemm_decode"
        output = folder / "attempt-0001"
        output.mkdir(parents=True)
        write_json(folder / "completed.json", dict(attempt=output.name))
        records = [
            dict(
                index=i,
                config=c,
                config_id=b200.config_id(c),
                status="benchmarked",
                latency_ms=10,
                post_compile=dict(
                    keep=True,
                    status="pass",
                    resources={
                        "kernel": dict(
                            spill_stores_bytes=0,
                            spill_loads_bytes=0,
                            local_bytes=0,
                        )
                    },
                ),
            )
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
                records[1]["post_compile"].update(keep=False, status="reject")
            elif failure == "benchmark":
                records[1]["status"] = "benchmark_error"
            write_json(output / "tiletune.json", dict(ranking=ranking, selection=dict(selected_indices=selected)))
        write_json(output / "outcomes.json", records)
    report = b200.oracle_audit(tmp_path, plan)
    assert report["complete"] and report["workloads_audited"] == 1
    assert len(report["oracles"]) == 3
    assert report["workloads_retained"] == (1 if failure is None else 0)
    assert (tmp_path / "oracle_retention.csv").is_file()


@pytest.mark.parametrize(
    "post_compile_status,post_compile_keep,retained",
    [("pass", True, True), ("unknown", True, False), ("reject", False, False)],
)
def test_oracle_audit_requires_post_compile_policy_pass(
    tmp_path, post_compile_status, post_compile_keep, retained
):
    plan = b200.study_plan(workloads=["gemm_decode"])
    configs = plan[0]["configs"][:8]
    for item in plan:
        folder = tmp_path / item["experiment"] / "gemm_decode"
        output = folder / "attempt-0001"
        output.mkdir(parents=True)
        write_json(folder / "completed.json", dict(attempt=output.name))
        records = [
            dict(index=i, config=c, config_id=b200.config_id(c), status="benchmarked", latency_ms=i + 1)
            for i, c in enumerate(configs)
        ]
        if item["experiment"] == "E3":
            records[0]["post_compile"] = dict(
                keep=post_compile_keep,
                status=post_compile_status,
                resources={
                    "kernel": dict(
                        spill_stores_bytes=12,
                        spill_loads_bytes=8,
                        local_bytes=8,
                    )
                },
            )
            ranking = [dict(index=i, score=i + 1, tier="eligible", tie_last_rank=i + 1) for i in range(8)]
            write_json(
                output / "tiletune.json",
                dict(ranking=ranking, selection=dict(selected_indices=[0, 1, 2, 3])),
            )
        write_json(output / "outcomes.json", records)
    report = b200.oracle_audit(tmp_path, plan)
    oracle = next(row for row in report["oracles"] if row["baseline"] == "E1")
    assert oracle["retained"] is retained
    assert oracle["post_compile_status"] == post_compile_status
    assert oracle["post_compile_counters_complete"]
    assert not oracle["zero_spill_local_observed"]


@pytest.mark.parametrize(
    "variant,gpu_count,reuse",
    [("baseline", 1, False), ("multi_gpu", 4, False), ("tiletune", 4, False), ("baseline", 1, True)],
)
def test_worker_wires_frozen_request_and_publishes_complete_outcomes(
    tmp_path, monkeypatch, variant, gpu_count, reuse
):
    """Exercise the actual worker/report path with a deterministic fake compiler."""
    from concurrent.futures import Future
    from contextlib import nullcontext
    from types import SimpleNamespace
    import torch
    import tilelang.autotuner as autotuner
    import tilelang.tiletune as tiletune
    from experiments.common import system, kernels
    from experiments.utils import cli

    configs = [dict(block=32), dict(block=64)]
    request = dict(
        workload=b200.study_plan()[0]["workload"],
        variant=variant,
        gpu_count=gpu_count,
        settings=dict(
            b200.SETTINGS,
            preflight=False,
            post_compile_resource_policy="reject" if variant == "tiletune" else "report",
        ),
        configs=configs,
        indices=[0, 1],
    )
    executed_configs = configs
    if reuse:
        source = tmp_path / "source"
        source.mkdir()
        old_records = [
            dict(
                index=0,
                original_index=0,
                config=configs[0],
                config_id=b200.config_id(configs[0]),
                status="benchmarked",
                latency_ms=0.5,
            ),
            dict(
                index=1,
                original_index=1,
                config=configs[1],
                config_id=b200.config_id(configs[1]),
                status="post_compile_rejected",
            ),
        ]
        write_json(source / "outcomes.json", old_records)
        write_json(source / "compilation.json", {"0": {"status": "compiled"}, "1": {"status": "post_compile_rejected"}})
        (source / "benchmarks.tsv").write_text(
            'index\tstatus\tlatency_ms\tconfig\terror\n0\tok\t0.5\t{"block": 32}\t\n'
        )
        (source / "resource-filter.tsv").write_text(
            "index\tstage\tverdict\treason\tconfig\tdetails\n"
            '0\tpost_compile\tkeep\tpassed\t{"block": 32}\t{}\n'
            '1\tpost_compile\treject\tspills\t{"block": 64}\t{}\n'
        )
        names = ["outcomes.json", "compilation.json", "benchmarks.tsv", "resource-filter.tsv"]
        request["reuse"] = dict(
            attempt=str(source),
            files={name: b200.file_hash(source / name) for name in names},
            reused_indices=[0],
            rerun_indices=[1],
            source_tuning_seconds=4,
            source_worker_wall_seconds=5,
        )
        executed_configs = configs[1:]
    write_json(tmp_path / "request.json", request)
    monkeypatch.setattr(system, "family_module", lambda *args: SimpleNamespace(get_configs=lambda: configs))
    monkeypatch.setattr(torch, "set_num_threads", lambda *args: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *args: None)
    monkeypatch.setattr(torch, "Generator", lambda **kwargs: SimpleNamespace(manual_seed=lambda seed: None))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: gpu_count)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    tensor = SimpleNamespace(to=lambda device: None)
    case = SimpleNamespace(
        build=lambda **kwargs: None,
        inputs=lambda *args: [tensor],
        check_input_values=lambda *args: None,
        reference=lambda *args: None,
        check=lambda *args: None,
        out_idx=[2],
        pass_configs={},
        input_values={},
        rtol=0.02,
        atol=0.02,
    )
    monkeypatch.setattr(kernels, "make_case", lambda w: case)
    monkeypatch.setattr(autotuner, "set_autotune_inputs", lambda *args: nullcontext())
    monkeypatch.setattr(tiletune, "current_target", lambda: dict(kind="cuda", arch="sm_100a"))
    monkeypatch.setattr(tiletune, "query_device_limits", lambda target: dict(registers_per_sm=65536))
    monkeypatch.setattr(cli, "device_info", lambda devices: devices)
    monkeypatch.setattr(cli, "source_hashes", lambda *args: {})
    monkeypatch.setattr(system.subprocess, "check_output", lambda *args, **kwargs: "test-version")

    class Tuner:
        def __init__(self, build, pool):
            assert pool == executed_configs
            self.configs = pool
            self.tiletune_session = None

        def set_compile_args(self, **kwargs):
            assert kwargs["execution_backend"] == "tvm_ffi"
            return self

        def set_profile_args(self, **kwargs):
            assert kwargs["manual_check_prog"] is case.check
            assert kwargs["backend"] == "event"
            return self

        def set_benchmark_report_path(self, path):
            self.benchmark_path = Path(path)
            return self

        def set_filter_args(self, config):
            assert config.enabled and config.action == "report"
            assert config.max_spills is config.max_local_size_bytes is None
            assert config.max_attention_spills is config.max_attention_local_size_bytes is None
            return self

        def set_tiletune_args(self, config):
            assert config.alpha == 0.5 and config.ranking_metric == "memory" and not config.memory_diagnostics
            assert config.mode == "report_only" and config.max_spill_bytes is config.max_local_bytes is None
            assert config.post_compile_policy == dict(mode="reject", max_spill_bytes=0, max_local_bytes=0)
            self.tiletune_report = dict(
                selection=dict(selected_indices=[0], pool_size=2, alpha=0.5, strict_budget=True),
                configs=[dict(index=i, config=c, status="benchmarked" if i == 0 else "not_selected") for i, c in enumerate(configs)],
            )
            self.tiletune_session = SimpleNamespace(selection=self.tiletune_report["selection"], finish=lambda: None)
            return self

        def _resolve_num_compile_workers(self):
            return 64

        def _prepare_compile_execution(self, *, config_indices):
            future = Future()
            future.set_result([(i, self.configs[i], None, None) for i in config_indices])
            return None, [future], {future: [(i, self.configs[i]) for i in config_indices]}, "test"

        def run(self, **kwargs):
            assert self._resolve_num_compile_workers() == 64
            assert kwargs["benchmark_multi_gpu"] == (gpu_count == 4)
            assert kwargs["use_pipeline"] == kwargs["enable_grouped_compile"] == (variant == "tiletune")
            assert kwargs["group_compile_size"] == 8 and not kwargs["early_stop"]
            selected = [0] if variant == "tiletune" or reuse else [0, 1]
            self._prepare_compile_execution(config_indices=selected)
            with self.benchmark_path.open("w") as stream:
                stream.write("index\tstatus\tlatency_ms\tconfig\terror\n")
                for i in selected:
                    stream.write(f"{i}\tok\t{i + 1}\t{json.dumps(self.configs[i])}\t\n")
            if reuse:
                (tmp_path / "continuation-resource-filter.tsv").write_text(
                    "index\tstage\tverdict\treason\tconfig\tdetails\n"
                    '0\tpost_compile\tkeep\tpassed\t{"block": 64}\t{}\n'
                )
            return SimpleNamespace(config=self.configs[0], latency=1)

    monkeypatch.setattr(autotuner, "AutoTuner", Tuner)
    system.worker(tmp_path / "request.json", tmp_path)
    experiment = b200.read(tmp_path / "experiment.json")
    assert experiment["configs"] == configs
    assert experiment["measurement"]["backend"] == "event"
    summary = b200.read(tmp_path / "summary.json")
    assert summary["config_count"] == 2 and summary["benchmark_gpu_count"] == gpu_count
    assert summary["compiler_workers"] == 64
    outcomes = b200.read(tmp_path / "outcomes.json")
    assert outcomes[0]["status"] == "benchmarked"
    assert outcomes[1]["status"] == ("not_selected" if variant == "tiletune" else "benchmarked")
    if reuse:
        assert outcomes[0]["reused"] and not outcomes[1]["reused"]
        assert summary["reused_candidate_count"] == summary["continuation_candidate_count"] == 1
        assert summary["tuning_seconds"] >= 4
        assert b200.read(tmp_path / "reuse.json")["rerun_indices"] == [1]
        assert any(line.startswith("1\t") for line in (tmp_path / "resource-filter.tsv").read_text().splitlines())
