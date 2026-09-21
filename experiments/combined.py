"""Run the fixed B200 TileTune + system-optimization experiment.

The brute-force side is read from immutable baseline bundles.  The live side
uses a strict memory-ranking fraction alpha (default 50%), post-compile spill/local-memory detection,
compile/benchmark pipelining, compile groups of four, and two benchmark GPUs.
Every brute-force winner is checked after TileTune freezes its selection and
again after compilation; oracle data never participates in ranking.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

from experiments.common.spec import Device, TARGETS, Workload, configuration_space
from experiments.families import FAMILIES, family_module
from experiments.utils.baseline_store import measurement_sources, verify_bundle
from experiments.utils.io import write_json


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = Path("/raid/ziren/tilelang/experiments/results/baselines/blackwell")
BUNDLES = {
    "flash_attention": BASELINE_ROOT / "flash_attention/fd3f49f83f81a28395daae3ecbea8c4325ee968d0a3bb252fb80ac363f8e0416",
    "gemm": BASELINE_ROOT / "gemm/aee7db3f46bb898c0b59619a65e4c929a9774d274996ad7e762ac9073cf6fbd8",
    "gemm_fp8": BASELINE_ROOT / "gemm_fp8/f91ec18a262b2067dd447224d7dd137dd180316ee9b419922b05f17e4dc5e3b8",
    "grouped_gemm": BASELINE_ROOT / "grouped_gemm/7f8fdccd7b07996aced4af486f123bffdecdc53ae474f16fc0ef1e86ddd898b9",
    "kda": BASELINE_ROOT / "kda/df6820b9e5da0e76f4c484c1dd7ed2418127f9b58491de19d70147db5811c3c5",
}
FAMILY_ORDER = ("gemm", "flash_attention", "kda", "gemm_fp8", "grouped_gemm")
OPS = {family: op for op, family in FAMILIES.items()}
SETTINGS = {
    "alpha": 0.5,
    "warmup": 10,
    "rep": 50,
    "timeout": 60,
    "workers": 4,
    "group_size": 4,
    "seed": 123,
    "pipeline": True,
    "multi_gpu": True,
    # Detect and report spill/local-memory use without turning either counter
    # into a rejection policy.  Physical launch/register limits remain strict.
    "max_spill_bytes": None,
    "max_local_bytes": None,
}


def _read(path):
    return json.loads(Path(path).read_text())


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _baseline_case(bundle, workload):
    return bundle / "collection" / "blackwell" / "test" / workload / "brute_force"


def _cases(family):
    return family_module(OPS[family], "cases").cases(holdout=True)


def _verify_baselines():
    device = Device("blackwell", TARGETS["blackwell"])
    report = {"allowed_source_difference": ["experiments/common/run.py"], "families": {}}
    for family in FAMILY_ORDER:
        bundle = BUNDLES[family]
        identity = _read(bundle / "identity.json")
        manifest = verify_bundle(bundle, identity)
        measurement = _read(bundle / "measurement.json")
        current_sources = measurement_sources([family])
        expected_sources = measurement["sources"]
        mismatches = sorted(
            name for name in set(current_sources) | set(expected_sources) if current_sources.get(name) != expected_sources.get(name)
        )
        if mismatches != ["experiments/common/run.py"]:
            raise ValueError(f"{family}: baseline measurement sources differ: {mismatches}")
        if measurement["timing"] != {**{k: SETTINGS[k] for k in ("warmup", "rep", "timeout", "workers")}, "case_timeout": 7200}:
            raise ValueError(f"{family}: archived timing protocol differs")
        cases = _cases(family)
        if {case.name for case in cases} != {case["name"] for case in identity["splits"]["test"]}:
            raise ValueError(f"{family}: archived and live workload sets differ")
        case_rows = {}
        for workload in cases:
            configs = configuration_space(workload, device)["configs"]
            if configs != identity["pools"][workload.name]:
                raise ValueError(f"{family}/{workload.name}: archived and live config pools differ")
            base = _baseline_case(bundle, workload.name)
            result = _read(base / "result.json")
            if result["status"] != "completed" or result["configs"] != len(configs):
                raise ValueError(f"{family}/{workload.name}: incomplete brute-force result")
            case_rows[workload.name] = {
                "configs": len(configs),
                "result": str((base / "result.json").resolve()),
                "oracle": str((base / "brute_force.json").resolve()),
                "tuning_seconds": result["tuning_seconds"],
                "winner": result["winner"],
            }
        report["families"][family] = {
            "bundle": str(bundle.resolve()),
            "bundle_manifest_sha256": _sha256(bundle / "complete.json"),
            "artifact_count": len(manifest["artifacts"]),
            "source_mismatches": mismatches,
            "cases": case_rows,
        }
    return report


def _worker(request_path, output):
    import torch
    import tilelang
    from tilelang.autotuner import AutoTuner, set_autotune_inputs
    from tilelang.tiletune import TileTuneConfig, current_target, query_device_limits
    from tiletune_core.ranking import alpha_budget
    from experiments.common.kernels import make_case
    from experiments.utils.cli import device_info, source_hashes
    from experiments.utils.results import config_key, load_oracle

    if not Path(tilelang.__file__).resolve().is_relative_to(ROOT):
        raise RuntimeError(f"worker imported TileLang from the wrong checkout: {tilelang.__file__}")
    request = _read(request_path)
    workload = Workload(**request["workload"])
    settings = request["settings"]
    output = Path(output)
    torch.cuda.set_device(0)
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"combined experiment requires exactly two visible GPUs, found {torch.cuda.device_count()}")
    if any(torch.cuda.get_device_name(index) != "NVIDIA B200" for index in range(2)):
        raise RuntimeError("combined experiment requires two NVIDIA B200 GPUs")
    torch.backends.cuda.matmul.allow_tf32 = False
    target = current_target()
    limits = query_device_limits(target)
    case = make_case(workload)
    configs = configuration_space(workload, Device("blackwell", TARGETS["blackwell"]))["configs"]
    requested_k = alpha_budget(len(configs), settings["alpha"])
    inputs = case.inputs("cuda:0", torch.Generator(device="cuda:0").manual_seed(settings["seed"]))
    case.check_input_values(inputs)
    references = {}
    for device in range(2):
        values = [value if value.device.index == device else value.to(f"cuda:{device}") for value in inputs]
        references[device] = case.reference(*values)
        torch.cuda.synchronize(device)

    guard = {}

    class GuardedTuner(AutoTuner):
        def _prepare_compile_execution(self, *args, **kwargs):
            # AutoTuner calls this only after prepare_top_k has frozen the full
            # ranking and selection.  Opening the oracle here cannot influence it.
            oracle = load_oracle(request["oracle_path"])
            oracle_key = config_key(oracle["winner"]["config"])
            oracle_index = next((i for i, config in enumerate(configs) if config_key(config) == oracle_key), None)
            selected = list(self.tiletune_session.selection["selected_indices"])
            guard.update(
                oracle_opened_after_selection=True,
                oracle_index=oracle_index,
                oracle_config=oracle["winner"]["config"],
                oracle_latency_ms=oracle["winner"]["latency_ms"],
                selected=oracle_index in selected if oracle_index is not None else False,
                requested_k=requested_k,
                selected_count=len(selected),
            )
            write_json(output / "oracle-selection-guard.json", guard)
            if not guard["selected"]:
                raise RuntimeError("frozen TileTune selection omitted the brute-force winner")
            return super()._prepare_compile_execution(*args, **kwargs)

    config = TileTuneConfig(
        enabled=True,
        mode="reject",
        ranking_metric="memory",
        alpha=settings["alpha"],
        input_values=case.input_values or None,
        device_limits=limits,
        report_path=str(output / "tiletune.json"),
        max_spill_bytes=settings["max_spill_bytes"],
        max_local_bytes=settings["max_local_bytes"],
    )
    write_json(
        output / "experiment.json",
        {
            **request,
            "target": target,
            "device_limits": limits,
            "devices": device_info([0, 1]),
            "config_count": len(configs),
            "requested_k": requested_k,
            "alpha": settings["alpha"],
            "strict_percentage_cutoff": True,
            "source_sha256": source_hashes("experiments/combined.py"),
            "cold_kernel_cache": True,
            "cold_autotune_cache": True,
        },
    )
    with set_autotune_inputs(inputs):
        tuner = (
            GuardedTuner(case.build, configs)
            .set_compile_args(
                out_idx=case.out_idx,
                pass_configs=case.pass_configs,
                target=target,
                execution_backend="tvm_ffi",
            )
            .set_profile_args(
                ref_prog=lambda *values: references[values[0].device.index],
                manual_check_prog=case.check,
                backend="event",
                cache_input_tensors=False,
                rtol=case.rtol,
                atol=case.atol,
                max_mismatched_ratio=0.0,
            )
            .set_tiletune_args(config)
            .set_benchmark_report_path(str(output / "benchmarks.tsv"))
        )
    started = time.perf_counter()
    result = tuner.run(
        warmup=settings["warmup"],
        rep=settings["rep"],
        timeout=settings["timeout"],
        early_stop=False,
        use_pipeline=True,
        enable_grouped_compile=True,
        group_compile_size=settings["group_size"],
        benchmark_multi_gpu=True,
        benchmark_devices=[0, 1],
    )
    tuning_seconds = time.perf_counter() - started
    report = tuner.tiletune_report
    oracle_record = report["configs"][guard["oracle_index"]]
    post_compile = oracle_record.get("post_compile")
    guard.update(
        post_compile_observed=post_compile is not None,
        post_compile_keep=bool(post_compile and post_compile.get("keep")),
        post_compile_status=post_compile.get("status") if post_compile else None,
        post_compile_classification=post_compile.get("classification") if post_compile else None,
        post_compile_reasons=post_compile.get("reasons") if post_compile else None,
        post_compile_resources=post_compile.get("resources") if post_compile else None,
        final_status=oracle_record["status"],
        benchmarked=oracle_record["status"] == "benchmarked",
    )
    guard["passed"] = guard["selected"] and guard["post_compile_keep"] and guard["benchmarked"]
    write_json(output / "oracle-guard.json", guard)
    summary = {
        "status": "completed" if guard["passed"] else "invalid",
        "family": request["family"],
        "workload": workload.name,
        "variant": "combined",
        "tuning_seconds": tuning_seconds,
        "config_count": len(configs),
        "requested_k": requested_k,
        "alpha": settings["alpha"],
        "selected_count": report["selection"]["selected_count"],
        "selection_budget_shortfall": report["selection"].get("shortfall", 0),
        "selection_budget_excess": report["selection"].get("budget_excess", 0),
        "winner_config": result.config,
        "winner_latency_ms": result.latency,
        "candidate_statuses": dict(Counter(row["status"] for row in report["configs"])),
        "oracle_guard": guard,
    }
    write_json(output / "summary.json", summary)
    if not guard["passed"]:
        raise RuntimeError("oracle winner did not survive selection, post-compile filtering, and benchmark validation")
    # Some long multi-GPU runs expose a native allocator fault during Python's
    # process-global CUDA/FFI teardown, after every benchmark thread has joined
    # and all result artifacts are durable.  Workers are deliberately isolated
    # processes, so exit directly only after the complete oracle-guarded result
    # has been serialized; the OS releases their CUDA contexts.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _attempt(case_root):
    attempts = case_root / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    completed = []
    for path in sorted(attempts.glob("*/summary.json")):
        summary = _read(path)
        monitor = path.parent / "monitor.json"
        clean = monitor.is_file() and _read(monitor).get("status") == "uncontended"
        if clean and summary.get("status") == "completed" and summary.get("oracle_guard", {}).get("passed"):
            completed.append((path.parent, summary))
    if completed:
        return completed[-1][0], completed[-1][1], True
    number = max([int(path.name) for path in attempts.iterdir() if path.is_dir() and path.name.isdigit()] or [0]) + 1
    output = attempts / f"{number:04d}"
    output.mkdir()
    return output, None, False


def _aggregate(root, verification):
    rows = []
    for family in FAMILY_ORDER:
        for workload in _cases(family):
            attempts = root / family / workload.name / "attempts"
            summaries = [(_read(path), path.parent) for path in sorted(attempts.glob("*/summary.json"))]
            valid = [
                (summary, path)
                for summary, path in summaries
                if (path / "monitor.json").is_file()
                and _read(path / "monitor.json").get("status") == "uncontended"
                and summary.get("status") == "completed"
                and summary["oracle_guard"]["passed"]
            ]
            if not valid:
                continue
            summary, path = valid[-1]
            baseline = verification["families"][family]["cases"][workload.name]
            rows.append(
                {
                    **summary,
                    "attempt": str(path.resolve()),
                    "brute_force_tuning_seconds": baseline["tuning_seconds"],
                    "end_to_end_speedup": baseline["tuning_seconds"] / summary["tuning_seconds"],
                    "brute_force_result": baseline["result"],
                }
            )
    families = {}
    for family in FAMILY_ORDER:
        family_rows = [row for row in rows if row["family"] == family]
        if not family_rows:
            continue
        speedups = [row["end_to_end_speedup"] for row in family_rows]
        families[family] = {
            "cases": len(family_rows),
            "mean_case_speedup": statistics.fmean(speedups),
            "geomean_case_speedup": math.exp(statistics.fmean(math.log(value) for value in speedups)),
            "median_case_speedup": statistics.median(speedups),
            "total_time_speedup": sum(row["brute_force_tuning_seconds"] for row in family_rows)
            / sum(row["tuning_seconds"] for row in family_rows),
            "all_oracle_winners_preserved": all(row["oracle_guard"]["passed"] for row in family_rows),
        }
    result = {"cases": rows, "families": families, "complete": len(rows) == 25}
    if rows:
        speedups = [row["end_to_end_speedup"] for row in rows]
        result["overall"] = {
            "cases": len(rows),
            "mean_case_speedup": statistics.fmean(speedups),
            "geomean_case_speedup": math.exp(statistics.fmean(math.log(value) for value in speedups)),
            "median_case_speedup": statistics.median(speedups),
            "total_time_speedup": sum(row["brute_force_tuning_seconds"] for row in rows) / sum(row["tuning_seconds"] for row in rows),
            "all_oracle_winners_preserved": all(row["oracle_guard"]["passed"] for row in rows),
        }
    write_json(root / "comparison.json", result)
    return result


def main(argv=None):
    if argv is None and len(sys.argv) == 4 and sys.argv[1] == "--worker":
        _worker(sys.argv[2], sys.argv[3])
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs=2, default=[2, 3])
    parser.add_argument("--alpha", type=float, default=SETTINGS["alpha"], help="strict config-pool fraction in (0, 1]; default 0.50")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.gpus)) != 2:
        parser.error("--gpus requires two distinct physical GPU indices")
    if not math.isfinite(args.alpha) or not 0 < args.alpha <= 1:
        parser.error("--alpha must be finite and in (0, 1]")
    settings = {**SETTINGS, "alpha": args.alpha}
    verification = _verify_baselines()
    from experiments.utils.monitor import run_monitored, snapshot, visible_gpus

    observed = snapshot()
    gpus = [gpu for gpu in visible_gpus(observed) if int(gpu["index"]) in args.gpus]
    if len(gpus) != 2 or {gpu["name"] for gpu in gpus} != {"NVIDIA B200"}:
        parser.error("requested physical devices must resolve to two NVIDIA B200 GPUs")
    root = (args.output or ROOT / "experiments/results" / ("b200-memory-combined-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))).resolve()
    if root.exists() and not args.resume:
        parser.error(f"output already exists: {root}; pass --resume")
    root.mkdir(parents=True, exist_ok=args.resume)
    write_json(root / "baseline-verification.json", verification)
    stable_gpus = [{key: gpu[key] for key in ("index", "uuid", "name")} for gpu in gpus]
    plan = {
        "version": 1,
        "settings": settings,
        "physical_gpus": stable_gpus,
        "families": {family: [case.to_dict() for case in _cases(family)] for family in FAMILY_ORDER},
        "oracle_policy": "opened only after frozen selection; selected + post-compile keep + benchmarked required",
    }
    plan_path = root / "plan.json"
    if plan_path.exists():
        existing = _read(plan_path)
        # Older interrupted attempts recorded volatile utilization/clock fields.
        existing["physical_gpus"] = [
            {key: gpu[key] for key in ("index", "uuid", "name")} for gpu in existing["physical_gpus"]
        ]
        if existing != plan:
            raise ValueError("resume plan differs from the existing experiment")
    write_json(plan_path, plan)
    for family in FAMILY_ORDER:
        for workload in _cases(family):
            output, _, done = _attempt(root / family / workload.name)
            if done:
                continue
            baseline = verification["families"][family]["cases"][workload.name]
            request = {
                "version": 1,
                "family": family,
                "workload": workload.to_dict(),
                "settings": settings,
                "oracle_path": baseline["oracle"],
                "brute_force_result": baseline["result"],
            }
            request_path = output / "request.json"
            write_json(request_path, request)
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=",".join(gpu["uuid"] for gpu in gpus),
                TILELANG_DISABLE_CACHE="1",
                TILELANG_AUTO_TUNING_DISABLE_CACHE="1",
                TILELANG_AUTO_TUNING_CPU_COUNTS=str(settings["workers"]),
                TILELANG_AUTOTUNE_TIMING_LOG=str(output / "timings.tsv"),
            )
            run_monitored(
                [sys.executable, "-m", "experiments.combined", "--worker", str(request_path), str(output)],
                output,
                gpus,
                env=env,
                cwd=ROOT,
                timeout=7200,
            )
            _aggregate(root, verification)
    comparison = _aggregate(root, verification)
    if not comparison["complete"] or not comparison["overall"]["all_oracle_winners_preserved"]:
        raise RuntimeError("combined experiment did not complete with all oracle winners preserved")
    print(json.dumps(comparison["families"], indent=2))
    print(json.dumps(comparison["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
