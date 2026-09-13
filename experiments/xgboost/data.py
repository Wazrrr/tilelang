"""Read measured candidates without admitting labels into prediction features."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def canonical_workload(workload):
    """Names and search grids do not distinguish training/test workloads."""
    from experiments.portable.spec import Workload

    w = workload if isinstance(workload, Workload) else Workload(**workload)
    defaults = {
        "gemm": dict(batch=1, transpose_a=False, transpose_b=False, epilogue="none"),
        "attention": dict(causal=False),
        "rmsnorm": dict(epsilon=1e-6),
    }
    parameters = {**defaults.get(w.op, {}), **w.parameters}
    if w.op == "rmsnorm":
        parameters["epsilon"] = float(parameters["epsilon"])
    return dict(op=w.op, dtype=w.dtype, parameters=parameters)


def workload_key(context):
    # Holding out a shape means holding out all its configurations, repeats,
    # aliases and implementations, including measurements on another device.
    return digest(context["workload"])


def domain(context):
    return {key: context[key] for key in ("implementation", "kernel_sha256", "device", "benchmark_backend", "environment")}


def make_context(workload, implementation, target, device_name, backend, source_hashes, *, environment=None):
    target = dict(target)
    kind = target["kind"]
    arch = target.get("mcpu" if kind == "hip" else "arch")
    if not arch or not device_name:
        raise ValueError("XGBoost needs an explicit architecture and observed device name")
    arch = arch.rstrip("af") if kind == "cuda" else arch.split(":")[0] if kind == "hip" else arch
    paths = ["experiments/portable/kernels.py"] if implementation.startswith("portable.") else [f"experiments/{implementation}/kernel.py"]
    if implementation in ("portable.attention", "flash_attention"):
        paths += ["examples/flash_attention/example_mha_fwd_bshd.py", "examples/flash_attention/example_mha_tiletune.py"]
    if implementation == "gemm_fp8":
        paths += ["examples/gemm_fp8/example_tilelang_gemm_fp8.py"]
    missing = set(paths) - source_hashes.keys()
    if missing:
        raise ValueError(f"missing kernel source fingerprints: {sorted(missing)}")
    return dict(
        workload=canonical_workload(workload),
        implementation=implementation,
        device=dict(kind=kind, arch=arch, name=device_name),
        benchmark_backend=backend,
        kernel_sha256={path: source_hashes[path] for path in paths},
        environment=environment,
    )


def context_from_experiment(experiment):
    if "workload" in experiment and "device" in experiment:
        observation = experiment.get("device_observation")
        if not observation:
            raise ValueError("training requires actual device observations")
        return make_context(
            experiment["workload"],
            "portable." + experiment["workload"]["op"],
            observation["target"],
            observation["name"],
            "event",
            experiment["source_sha256"],
            environment=dict(
                native_build=experiment.get("native_build"),
                torch_version=observation.get("torch_version"),
                runtime_version=observation.get("runtime_version"),
            ),
        )
    args = experiment["arguments"]
    if "sequence" in args:
        implementation = "flash_attention"
        workload = dict(
            name="attention",
            op="attention",
            dtype="float16",
            parameters={
                **{key: args[key] for key in ("batch", "heads", "sequence", "dim")},
                "causal": args["causal"],
            },
        )
    else:
        implementation = "gemm_fp8" if args["dtype"].startswith("float8") else "gemm"
        workload = dict(
            name="gemm",
            op="gemm",
            dtype=args["dtype"],
            parameters={
                **{key: args[key] for key in ("m", "n", "k")},
                "transpose_b": True,
            },
        )
    return make_context(
        workload,
        implementation,
        experiment["target"],
        experiment["devices"][0]["name"],
        args["backend"],
        experiment["source_sha256"],
        environment=dict(
            native_build=experiment.get("native_build"),
            torch_version=experiment.get("torch_version"),
            runtime_version=experiment.get("runtime_version"),
        ),
    )


def features(context, config):
    """Only inputs known before compilation; no timings, ranks or counters."""
    result = {}

    def flatten(prefix, value):
        if isinstance(value, dict):
            for key, item in sorted(value.items()):
                flatten(prefix + "." + key, item)
        elif value is None:
            return
        elif isinstance(value, bool):
            result[prefix] = int(value)
        elif isinstance(value, int | float):
            if not math.isfinite(value):
                raise ValueError(f"nonfinite feature {prefix}")
            result[prefix] = value
        elif isinstance(value, str):
            result[prefix] = value
        elif isinstance(value, list):
            # Ordered compiler flag lists are categorical configuration inputs.
            result[prefix] = json.dumps(value, sort_keys=True, allow_nan=False)
        else:
            raise ValueError(f"unsupported feature {prefix}: {type(value).__name__}")

    flatten("workload", context["workload"])
    flatten("config", config)
    flatten("device", context["device"])
    result["implementation"] = context["implementation"]
    # Otherwise mixed compiler revisions or timing backends would produce
    # identical feature rows with different labels.
    result["execution_domain"] = digest(domain(context))
    return result


def read_runs(paths):
    """Read explicit exhaustive run directories (or their parent directories)."""
    manifests = set()
    for item in paths:
        path = Path(item).resolve()
        found = [path] if path.is_file() and path.name == "experiment.json" else list(path.rglob("experiment.json"))
        if not found:
            raise ValueError(f"no experiment.json under {path}")
        manifests.update(found)
    runs = []
    for path in sorted(manifests):
        experiment = json.loads(path.read_text())
        method = experiment.get("method", experiment.get("settings", {}).get("method"))
        if method not in ("brute_force", "exhaustive"):
            raise ValueError(f"{path}: expected an exhaustive run, found {method!r}")
        if not experiment.get("native_build"):
            raise ValueError(f"{path}: missing compiler build fingerprint; recollect with the current runner")
        summary_path = path.parent / ("result.json" if "settings" in experiment else "summary.json")
        if not summary_path.exists():
            raise ValueError(f"{path}: missing completed run marker")
        summary = json.loads(summary_path.read_text())
        if summary.get("status") not in ("completed", "ok"):
            raise ValueError(f"{summary_path}: run did not complete successfully")
        if "settings" in experiment and summary.get("correctness") != "passed":
            raise ValueError(f"{summary_path}: missing passed correctness check")
        record_path = path.parent / "outcomes.json"
        if record_path.exists():
            records = json.loads(record_path.read_text())
        else:
            record_path = path.parent / "tiletune.json"
            records = json.loads(record_path.read_text())["configs"]
        configs = experiment["configs"]
        if len(records) != len(configs) or {r["index"] for r in records} != set(range(len(configs))):
            raise ValueError(f"{record_path}: incomplete or duplicate candidate indices")
        successes = []
        terminal = {
            "benchmarked",
            "compilation_failed",
            "elaboration_failed",
            "analysis_failed",
            "pre_lowering_rejected",
            "post_compile_rejected",
            "benchmark_error",
            "benchmark_timeout",
        }
        for record in records:
            if record.get("status") not in terminal:
                raise ValueError(f"{record_path}: incomplete candidate outcome {record.get('status')!r}")
            if record.get("config") != configs[record["index"]]:
                raise ValueError(f"{record_path}: candidate configuration mismatch")
            if record["status"] != "benchmarked":
                continue
            latency = record.get("latency_ms")
            if isinstance(latency, bool) or not isinstance(latency, int | float) or not math.isfinite(latency) or latency <= 0:
                raise ValueError(f"{record_path}: invalid successful latency")
            successes.append(dict(index=record["index"], config=record["config"], latency_ms=latency))
        if not successes:
            raise ValueError(f"{record_path}: no correct, successfully measured candidates")
        runs.append(
            dict(
                context=context_from_experiment(experiment),
                configs=configs,
                samples=successes,
                provenance=dict(
                    experiment=str(path),
                    experiment_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    outcomes_sha256=hashlib.sha256(record_path.read_bytes()).hexdigest(),
                    statuses=dict(Counter(r["status"] for r in records)),
                    collection_seconds=summary.get("tuning_seconds"),
                ),
            )
        )
    return runs
