"""Readable intermediate snapshots must preserve analysis and capture mutations."""

from concurrent.futures import ThreadPoolExecutor
import json
import re
from threading import Barrier
from types import SimpleNamespace

import pytest

from tilelang.new_carver import CarverConfig, analyze_prim_func
from tilelang.new_carver.runtime import CarverSession
from tilelang.new_carver.trace import AnalysisTrace
from test_analysis import gemm
from test_cost import LIMITS
from test_modules import PROFILE, TARGET, attention


def checkpoint(text, name):
    match = re.search(r"^\[\d+\] " + re.escape(name) + r" \([^\n]+\)\n", text, re.MULTILINE)
    assert match is not None, name
    return json.JSONDecoder().raw_decode(text[match.end() :])[0]


@pytest.mark.parametrize("factory", [lambda: gemm(stages=2), attention])
def test_trace_captures_intermediate_data_without_changing_analysis(tmp_path, factory):
    func = factory()
    script = func.script()
    settings = dict(ranking_metric="pipeline_time", performance_model=PROFILE)
    before = analyze_prim_func(func, settings, target=TARGET, device_limits=LIMITS)
    path = tmp_path / "trace.log"
    after = analyze_prim_func(func, dict(settings, trace_path=str(path)), target=TARGET, device_limits=LIMITS)
    assert before == after
    assert func.script() == script
    text = path.read_text()
    assert "trace_error" not in text
    assert script in text
    col = checkpoint(text, "col")
    assert col["type"] == "_Collector"
    assert any(op["dependencies"] for op in col["operations"])
    assert all(not op["demands"] for op in col["operations"])  # snapshotted before propagation mutates them
    matrix = next(op for op in col["operations"] if "cRegion" in op["metadata"]["fields"])
    assert matrix["metadata"]["fields"]["c"]["scope"] == "local.fragment"
    assert matrix["metadata"]["fields"]["cRegion"]["buffer_id"] == matrix["metadata"]["fields"]["c"]["buffer_id"]
    propagated = checkpoint(text, "tile_propagation")
    assert propagated["type"] == "PropagationResult"
    assert any(op["demands"] for op in propagated["operation_demands"])
    assert len(propagated["input_loops"]) == len(propagated["per_iteration_inputs"])
    assert propagated["per_iteration_inputs"]
    assert not re.search(r"^\[\d+\] propagation ", text, re.MULTILINE)
    assert "propagation" not in after
    assert {k: propagated[k] for k in after["tile_propagation"]} == after["tile_propagation"]
    assert checkpoint(text, "specialization")["name"] == after["specialization"]["name"]
    assert checkpoint(text, "pressure.accumulator")["modeled_lower_bound"] == after["pressure"]["modeled_lower_bound"]
    assert checkpoint(text, "pressure.register_policy")["decision"] == after["pressure"]["decision"]
    assert checkpoint(text, "pipeline")["timing"] == after["modules"]["pipeline_overlap"]["timing"]
    assert checkpoint(text, "ranking")["score"] == checkpoint(text, "tile_cost")["score"] == after["tile_cost"]["score"]
    assert "analysis.py:" in text and "engine.py:" in text


def test_disabled_trace_never_constructs_snapshots(monkeypatch):
    calls = []
    monkeypatch.setattr("tilelang.new_carver.trace.collector_snapshot", lambda col: calls.append(col))
    monkeypatch.setattr("tilelang.new_carver.trace.propagation_snapshot", lambda result: calls.append(result))
    analyze_prim_func(gemm(), target=TARGET, device_limits=LIMITS)
    assert not calls


def test_trace_write_failure_does_not_change_decisions(tmp_path, capfd):
    func = gemm(stages=2)
    before = analyze_prim_func(func, target=TARGET, device_limits=LIMITS)
    # Opening an existing directory as a log file fails on every platform.
    after = analyze_prim_func(func, {"trace_path": str(tmp_path)}, target=TARGET, device_limits=LIMITS)
    assert before == after
    assert "Cannot write New Carver trace" in capfd.readouterr().out


def test_failed_analysis_preserves_earlier_checkpoints(tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError("test pressure failure")

    monkeypatch.setattr("tilelang.new_carver.analysis._pressure", fail)
    path = tmp_path / "failure.log"
    with pytest.raises(RuntimeError, match="test pressure failure"):
        analyze_prim_func(gemm(), {"trace_path": str(path)}, target=TARGET)
    text = path.read_text()
    assert checkpoint(text, "col")["operations"]
    assert checkpoint(text, "analysis_error")["message"] == "test pressure failure"
    assert "===== FAILED =====" in text


def test_session_trace_labels_candidates_and_appends(tmp_path):
    path = tmp_path / "session.log"
    configs = [{"stages": 0}, {"stages": 2}]
    session = CarverSession(CarverConfig(trace_path=str(path)), configs, target=TARGET, device_limits=LIMITS)
    for index, settings in enumerate(configs):
        session.elaborate(index, settings, gemm)
    blocks = path.read_text().split("===== New Carver analysis ")[1:]
    assert len(blocks) == 2
    for index, block in enumerate(blocks):
        assert checkpoint(block, "inputs")["trace_context"] == {"config_index": index, "config": configs[index]}


def test_parallel_traces_keep_analysis_blocks_together(tmp_path):
    path = tmp_path / "threads.log"
    barrier = Barrier(4)

    def write(index):
        with AnalysisTrace(str(path)) as trace:
            trace.record("start", lambda: {"index": index})
            barrier.wait(timeout=10)
            trace.record("end", lambda: {"index": index})

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, range(4)))
    blocks = path.read_text().split("===== New Carver analysis ")[1:]
    assert len(blocks) == 4
    assert {checkpoint(block, "start")["index"] for block in blocks} == set(range(4))
    for block in blocks:
        assert checkpoint(block, "start") == checkpoint(block, "end")
        assert block.count("===== COMPLETE =====") == 1


@pytest.mark.parametrize("value", ["", " ", False, 123])
def test_invalid_trace_path(value):
    with pytest.raises(ValueError, match="trace_path"):
        CarverConfig(trace_path=value)


def test_trace_path_does_not_change_cache_identity():
    assert CarverConfig(trace_path="trace.log").to_cache_key_dict() == CarverConfig().to_cache_key_dict()


def test_decorator_repeats_analysis_when_trace_is_requested(tmp_path, monkeypatch):
    import tilelang
    from tilelang.autotuner import AutoTuner

    path = tmp_path / "decorator.log"
    calls = []
    sentinel = object()

    def run(self, *args, **kwargs):
        func = self.jit_elaborate(**self.configs[0])
        analyze_prim_func(func, self.carver_args, target=TARGET, device_limits=LIMITS)
        calls.append(True)
        return SimpleNamespace(kernel=sentinel, config={})

    monkeypatch.setattr(AutoTuner, "run", run)
    tuned = tilelang.autotune(configs=[{"stages": 2}], carver={"trace_path": str(path)})(
        tilelang.jit(out_idx=[2], execution_backend="tvm_ffi")(gemm)
    )
    assert tuned() is sentinel
    assert tuned() is sentinel
    assert len(calls) == path.read_text().count("===== COMPLETE =====") == 2


@pytest.mark.parametrize("trace_enabled", [False, True])
def test_trace_bypasses_autotuner_result_cache(tmp_path, monkeypatch, trace_enabled):
    from tilelang.autotuner import AutoTuner
    from tilelang.autotuner.tuner import env

    tuner = AutoTuner(gemm, [{}]).set_compile_args(target=TARGET, execution_backend="tvm_ffi")
    tuner.set_carver_args(True, device_limits=LIMITS, trace_path=str(tmp_path / "trace.log") if trace_enabled else None)
    sentinel = SimpleNamespace(func=None)
    monkeypatch.setattr(tuner, "generate_cache_key", lambda *args: "trace-cache-test")
    monkeypatch.setitem(tuner._memory_cache, "trace-cache-test", sentinel)
    monkeypatch.setattr(env, "is_cache_enabled", lambda: True)
    monkeypatch.setattr(env, "is_autotune_cache_disabled", lambda: False)

    def uncached():
        raise RuntimeError("reached fresh analysis path")

    monkeypatch.setattr(tuner, "_ensure_jit_functions", uncached)
    if trace_enabled:
        with pytest.raises(RuntimeError, match="reached fresh analysis path"):
            tuner.run()
    else:
        assert tuner.run() is sentinel
