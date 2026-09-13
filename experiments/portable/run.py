"""Run or plan a workload × architecture matrix.

python -m experiments.portable.run --plan --smoke
python -m experiments.portable.run --devices hopper --smoke --method exhaustive

An external worker receives two absolute paths: request.json and result.json.
It runs in its own compiler environment and must return the request hash. There
is no implicit SSH connection, hardware substitution, or fallback ranking metric.
"""

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .spec import Device, TARGETS, Workload, configurations, default_workloads, load_manifest, support_reason


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")
    temporary.replace(path)


def make_request(workload, device, settings):
    settings = dict(settings)
    if settings.get("method") == "xgboost" and not settings.get("xgb_model_sha256"):
        path = settings.get("xgb_model")
        if not path:
            raise ValueError("xgboost requires an explicit trained --xgb-model")
        settings["xgb_model"] = str(Path(path).resolve())
        settings["xgb_model_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    payload = dict(version=1, workload=workload.to_dict(), device=device.to_dict(), settings=settings)
    return dict(payload, request_id=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())


def validate_request(request):
    if set(request) != {"version", "workload", "device", "settings", "request_id"} or request["version"] != 1:
        raise ValueError("unsupported worker request schema")
    workload, device = Workload(**request["workload"]), Device(**request["device"])
    if make_request(workload, device, request["settings"])["request_id"] != request["request_id"]:
        raise ValueError("worker request hash mismatch")
    return workload, device


def validate_result(result, request):
    if result.get("version") != 1 or result.get("request_id") != request["request_id"]:
        raise ValueError("worker result does not match this request")
    if result.get("status") not in ("completed", "analyzed", "unsupported", "unavailable", "model_unavailable", "failed"):
        raise ValueError("unknown worker result status")
    if result.get("workload") != request["workload"]["name"] or result.get("device") != request["device"]["name"]:
        raise ValueError("worker result workload/device identity does not match this request")
    if result["status"] == "completed":
        if request["settings"]["method"] == "analyze":
            raise ValueError("an analysis-only request cannot return a measured winner")
        latency = (result.get("winner") or {}).get("latency_ms")
        if isinstance(latency, bool) or not isinstance(latency, int | float) or not math.isfinite(latency) or latency <= 0:
            raise ValueError("completed worker result requires a finite positive winner latency")
        if result.get("correctness") != "passed" or not result.get("device_observation"):
            raise ValueError("completed worker result requires correctness and actual device observations")
        observation = result["device_observation"]
        if not observation.get("name") or not targets_match(request["device"]["target"], observation.get("target", {})):
            raise ValueError("worker observed a different target than requested")
        if request["settings"]["method"] == "xgboost":
            if result.get("model_sha256") != request["settings"]["xgb_model_sha256"]:
                raise ValueError("worker result used a different XGBoost model")
            selection = result.get("selection") or {}
            selected = selection.get("selected_indices")
            k = request["settings"]["top_k"]
            if (
                not isinstance(selected, list)
                or not selected
                or any(type(i) is not int or i < 0 for i in selected)
                or len(set(selected)) != len(selected)
                or len(selected) > k
                or selection.get("requested_k") != k
                or selection.get("selected_count") != len(selected)
                or result["winner"].get("index") not in selected
            ):
                raise ValueError("worker result violates the frozen XGBoost selection budget")
    elif result["status"] != "analyzed" and not result.get("reason"):
        raise ValueError("unsuccessful worker result requires a reason")
    return result


def targets_match(requested, observed):
    """Match runtime architectures, allowing CUDA compiler feature suffixes."""
    kind = requested.get("kind")
    if kind != observed.get("kind"):
        return False
    key = "mcpu" if kind == "hip" else "arch"
    left, right = requested.get(key), observed.get(key)
    if not left or not right:
        return False
    if kind == "cuda":
        return left.rstrip("af") == right.rstrip("af")
    if kind == "hip":
        return left.split(":")[0] == right.split(":")[0]
    return left == right


def run_external(request, device, output):
    request_path, result_path = output / "request.json", output / "result.json"
    write_json(request_path, request)
    started = time.perf_counter()
    with (output / "worker.log").open("w") as log:
        process = subprocess.Popen(
            [*device.worker, str(request_path.resolve()), str(result_path.resolve())],
            cwd=device.worker_cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process.wait(timeout=request["settings"]["case_timeout"])
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    if process.returncode:
        raise RuntimeError(f"worker exited with code {process.returncode}; see worker.log")
    result = validate_result(json.loads(result_path.read_text()), request)
    result["worker_wall_seconds"] = time.perf_counter() - started
    return result


def run_native(request, output):
    workload, device = validate_request(request)
    settings = request["settings"]
    result = dict(version=1, request_id=request["request_id"], workload=workload.name, device=device.name)
    reason = support_reason(workload, Device(device.name, device.target))
    if reason:
        return dict(result, status="unsupported", reason=reason)

    import torch
    from tilelang.tiletune import TileTuneConfig, current_target, query_device_limits, resolve_target
    from tilelang.tiletune.runtime import TileTuneSession
    from tilelang.cache.kernel_cache import KernelCache
    from experiments._common import source_hashes
    from .kernels import make_case

    target = resolve_target(device.target)
    limits = device.device_limits
    if settings["method"] != "analyze":
        try:
            actual = resolve_target(current_target())
        except ValueError as error:
            return dict(result, status="unavailable", reason=str(error))
        if not targets_match(device.target, actual.compiler_target()):
            return dict(
                result, status="unavailable", reason=f"requested {target.kind}:{target.arch}; visible device is {actual.kind}:{actual.arch}"
            )
        prop = torch.cuda.get_device_properties(torch.cuda.current_device())
        if device.name == "mi308" and "MI308" not in prop.name.upper():
            return dict(result, status="unavailable", reason=f"MI308 requested but visible device is {prop.name}")
        result["device_observation"] = dict(
            name=prop.name,
            target=actual.compiler_target(),
            torch_version=torch.__version__,
            runtime_version=torch.version.hip or torch.version.cuda,
        )
        if limits is None:
            limits = query_device_limits(device.target)
        torch.backends.cuda.matmul.allow_tf32 = False

    case = make_case(workload)
    performance_model = device.performance_model if settings["method"] != "xgboost" else None
    profile_identity = None
    if device.profiles and settings["method"] != "xgboost":
        from tilelang.tiletune import load_device_profile

        profile_path = device.profiles.get(workload.dtype)
        if profile_path is None and settings["metric"] == "pipeline_time":
            reason = f"no primitive profile for dtype {workload.dtype}; supplied profile dtypes: {sorted(device.profiles)}"
            if settings["method"] == "top_k":
                return dict(result, status="model_unavailable", reason=reason)
            result["model_unknown_reason"] = reason
        if profile_path is not None:
            path = Path(profile_path)
            bundle = json.loads(path.read_text())
            profile_identity = bundle["identity"]
            if settings["method"] != "analyze":
                observed = result["device_observation"]
                if (
                    profile_identity.get("device_name") != observed["name"]
                    or profile_identity.get("sm_count") != prop.multi_processor_count
                ):
                    raise ValueError("profile device identity does not match the visible GPU")
            performance_model = load_device_profile(path, input_dtype=workload.dtype, memory_regime=settings["memory_regime"])
    configs = configurations(workload, device)
    indices = settings.get("config_indices")
    if indices is not None:
        from experiments._common import select_configs

        indices, configs = select_configs(configs, indices)
    else:
        indices = list(range(len(configs)))
    if settings["method"] != "xgboost":
        config = TileTuneConfig(
            enabled=True,
            mode="report_only",
            ranking_metric=settings["metric"],
            top_k=settings["top_k"] if settings["method"] == "top_k" else None,
            performance_model=performance_model,
            device_limits=limits,
            report_path=str(output / "tiletune.json"),
            trace_path=str(output / "trace.log") if settings["trace"] else None,
            max_spill_bytes=None,
            max_local_bytes=None,
        )
    extra_sources = (
        ("examples/flash_attention/example_mha_fwd_bshd.py", "examples/flash_attention/example_mha_tiletune.py")
        if workload.op == "attention"
        else ()
    )
    hashes = source_hashes("experiments/portable/kernels.py", *extra_sources)
    xgb_report = None
    native_build = KernelCache._get_tilelang_lib_stamp()
    if settings["method"] == "xgboost":
        from experiments.xgboost.data import make_context
        from experiments.xgboost.model import Predictor

        started = time.perf_counter()
        context = make_context(
            workload,
            "portable." + workload.op,
            device.target,
            result["device_observation"]["name"],
            "event",
            hashes,
            environment=dict(
                native_build=native_build,
                torch_version=result["device_observation"]["torch_version"],
                runtime_version=result["device_observation"]["runtime_version"],
            ),
        )
        predictor = Predictor(settings["xgb_model"], expected_sha256=settings["xgb_model_sha256"], workers=settings["workers"])
        xgb_report = predictor.rank(context, configs, settings["top_k"])
        xgb_report["selection"]["wall_time_ms"] = (time.perf_counter() - started) * 1000
    write_json(
        output / "experiment.json",
        dict(
            **request,
            original_indices=indices,
            configs=configs,
            device_limits=limits,
            source_sha256=hashes,
            native_build=native_build,
            device_observation=result.get("device_observation"),
            performance_model=performance_model,
            profile_identity=profile_identity,
            note="Primitive rates are supplied before candidate measurements. Analysis-only mode performs no hardware queries.",
        ),
    )
    if settings["method"] == "analyze":
        session = TileTuneSession(config, configs, target=device.target, device_limits=limits)
        for idx, candidate in enumerate(configs):
            kwargs = {key: value for key, value in candidate.items() if key != "pass_configs"}
            passes = {**(case.pass_configs or {}), **(candidate.get("pass_configs") or {})}
            try:
                session.elaborate(idx, kwargs, case.build, pass_configs=passes)
            except Exception:
                # The batch boundary retains the precise failure in the record.
                continue
        report = session.finish()
        failures = sum(r["status"] in ("analysis_failed", "elaboration_failed") for r in report["configs"])
        return dict(
            result,
            status="analyzed" if not failures else "failed",
            reason=f"{failures} candidate analyses failed" if failures else None,
            configs=len(configs),
            scored=sum(e["tier"] == "eligible" for e in report["ranking"]),
            candidate_statuses=dict(Counter(r["status"] for r in report["configs"])),
        )

    from tilelang.autotuner import AutoTuner

    generator = torch.Generator(device="cuda").manual_seed(settings["seed"])
    inputs = case.inputs("cuda", generator)
    expected = case.reference(*inputs)
    if xgb_report is not None:
        from experiments.xgboost.execution import run_selected

        return dict(result, **run_selected(case, configs, indices, device.target, inputs, expected, settings, output, xgb_report))
    tuner = (
        AutoTuner(case.build, configs)
        .set_compile_args(target=device.target, execution_backend="tvm_ffi", out_idx=case.out_idx, pass_configs=case.pass_configs)
        .set_profile_args(
            supply_prog=lambda params: inputs,
            ref_prog=lambda *args: expected,
            manual_check_prog=case.check,
            backend="event",
            rtol=case.rtol,
            atol=case.atol,
            max_mismatched_ratio=0.0,
        )
        .set_tiletune_args(config)
        .set_benchmark_report_path(str(output / "benchmarks.tsv"))
    )
    started = time.perf_counter()
    try:
        winner = tuner.run(warmup=settings["warmup"], rep=settings["rep"], timeout=settings["timeout"], early_stop=False)
    except RuntimeError as error:
        if str(error).startswith("TileTune top_k selected no eligible"):
            return dict(result, status="model_unavailable", reason=str(error), configs=len(configs))
        raise
    report = tuner.tiletune_report
    from experiments._tiletune import winner_summary

    chosen = winner_summary(report, winner.config, winner.latency)
    chosen["original_index"] = indices[chosen["index"]]
    chosen.pop("score_cycles")  # Metric units are explicit below.
    return dict(
        result,
        status="completed",
        correctness="passed",
        winner=chosen,
        metric=settings["metric"],
        score_units="cycles" if settings["metric"] == "pipeline_time" else "byte-waves",
        tuning_seconds=time.perf_counter() - started,
        configs=len(configs),
        selection=report["selection"],
        candidate_statuses=dict(Counter(r["status"] for r in report["configs"])),
    )


def run_case(request, output):
    workload, device = validate_request(request)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "request.json", request)
    os.environ["TILELANG_AUTOTUNE_TIMING_LOG"] = str(output / "timings.tsv")
    try:
        # Every case owns a process and CUDA/HIP context. Timeouts or device
        # faults in one kernel cannot corrupt the following matrix entries.
        worker_device = (
            device
            if device.worker
            else replace(
                device,
                worker=[sys.executable, "-m", "experiments.portable.run", "--worker"],
                worker_cwd=str(Path(__file__).resolve().parents[2]),
            )
        )
        result = run_external(request, worker_device, output)
    except Exception as error:
        import traceback

        (output / "error.log").write_text(traceback.format_exc())
        result = dict(
            version=1,
            request_id=request["request_id"],
            workload=workload.name,
            device=device.name,
            status="failed",
            reason=f"{type(error).__name__}: {error}",
        )
    write_json(output / "result.json", result)
    return result


def worker_main(request_path, result_path):
    request = json.loads(Path(request_path).read_text())
    workload, device = validate_request(request)
    output = Path(result_path).resolve().parent
    os.environ["TILELANG_DISABLE_CACHE"] = "1"
    os.environ["TILELANG_AUTO_TUNING_DISABLE_CACHE"] = "1"
    os.environ["TILELANG_AUTO_TUNING_CPU_COUNTS"] = str(request["settings"]["workers"])
    os.environ["TILELANG_AUTOTUNE_TIMING_LOG"] = str(output / "timings.tsv")
    try:
        result = run_native(request, output)
    except Exception as error:
        import traceback

        (output / "error.log").write_text(traceback.format_exc())
        result = dict(
            version=1,
            request_id=request["request_id"],
            workload=workload.name,
            device=device.name,
            status="failed",
            reason=f"{type(error).__name__}: {error}",
        )
    write_json(result_path, result)
    return 0


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        return worker_main(sys.argv[2], sys.argv[3])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--devices", nargs="+", help=f"Device names in the manifest, or presets: {', '.join(TARGETS)}")
    parser.add_argument("--workloads", nargs="+")
    parser.add_argument("--smoke", action="store_true", help="Use smaller problem sizes; retain the supplied configuration grid")
    parser.add_argument(
        "--plan", action="store_true", help="Print the manifest and coverage without importing TileLang or querying a device"
    )
    parser.add_argument("--method", choices=["analyze", "exhaustive", "top_k", "xgboost"], default="analyze")
    parser.add_argument("--xgb-model", type=Path, help="Frozen model from python -m experiments.xgboost train")
    parser.add_argument("--metric", choices=["traffic_waves", "pipeline_time"], default="traffic_waves")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--memory-regime", choices=["cached", "streaming"], default="streaming")
    parser.add_argument("--config-indices", nargs="+", type=int)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/portable"))
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--case-timeout", type=int, default=3600)
    args = parser.parse_args()
    if args.method == "xgboost" and args.xgb_model is None:
        parser.error("--method xgboost requires --xgb-model")
    if any(getattr(args, key) <= 0 for key in ("top_k", "workers", "warmup", "rep", "timeout", "case_timeout")):
        parser.error("budgets, workers, repetitions, and timeouts must be positive")
    if args.manifest:
        devices, workloads = load_manifest(json.loads(args.manifest.read_text()))
        devices = [
            replace(d, profiles={dtype: str((args.manifest.resolve().parent / path).resolve()) for dtype, path in d.profiles.items()})
            if d.profiles
            else d
            for d in devices
        ]
    else:
        if args.devices and set(args.devices) - TARGETS.keys():
            parser.error(f"unknown target presets: {sorted(set(args.devices) - TARGETS.keys())}")
        devices, workloads = [Device(name, dict(TARGETS[name])) for name in (args.devices or TARGETS)], default_workloads(args.smoke)
    if args.manifest and args.devices:
        missing = set(args.devices) - {d.name for d in devices}
        if missing:
            parser.error(f"unknown manifest devices: {sorted(missing)}")
        devices = [d for d in devices if d.name in args.devices]
    if args.workloads:
        missing = set(args.workloads) - {w.name for w in workloads}
        if missing:
            parser.error(f"unknown workload names: {sorted(missing)}")
        workloads = [w for w in workloads if w.name in args.workloads]
    manifest = dict(version=1, devices=[d.to_dict() for d in devices], workloads=[w.to_dict() for w in workloads])
    if args.plan:
        print(
            json.dumps(
                dict(
                    manifest=manifest,
                    coverage=[
                        dict(
                            device=d.name,
                            workload=w.name,
                            configs=len(configurations(w, d)),
                            execution="external" if d.worker else "native" if not support_reason(w, d) else "unsupported",
                            reason=support_reason(w, d),
                            profile_supplied=d.performance_model is not None or w.dtype in (d.profiles or {}),
                        )
                        for d in devices
                        for w in workloads
                    ],
                ),
                indent=2,
            )
        )
        return 0
    root = args.output.resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "manifest.json", manifest)
    print(f"Results directory: {root}", flush=True)
    os.environ["TILELANG_DISABLE_CACHE"] = "1"
    os.environ["TILELANG_AUTO_TUNING_DISABLE_CACHE"] = "1"
    os.environ["TILELANG_AUTO_TUNING_CPU_COUNTS"] = str(args.workers)
    settings = {
        key: getattr(args, key)
        for key in (
            "method",
            "metric",
            "memory_regime",
            "top_k",
            "config_indices",
            "trace",
            "seed",
            "workers",
            "warmup",
            "rep",
            "timeout",
            "case_timeout",
        )
    }
    results = []
    if args.xgb_model is not None:
        settings["xgb_model"] = str(args.xgb_model.resolve())
        settings["xgb_model_sha256"] = hashlib.sha256(args.xgb_model.read_bytes()).hexdigest()
    for device in devices:
        for workload in workloads:
            request = make_request(workload, device, settings)
            result = run_case(request, root / device.name / workload.name)
            results.append(result)
            write_json(root / "summary.json", dict(version=1, results=results))
            print(f"{device.name}/{workload.name}: {result['status']} {result.get('reason') or ''}", flush=True)
    return int(any(result["status"] == "failed" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
