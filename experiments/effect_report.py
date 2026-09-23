"""Compare Carver, XGBoost, and TileTune against completed E1 oracles.

All methods rank the same held-out configuration pool and use an exact K with
original-index tie breaking.  Oracle outcomes are consulted only after each
ranking is frozen.  Online GPU cost is a counterfactual estimate obtained by
scaling the source one-GPU exhaustive tuning time by attempted shortlist size.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time

from experiments.common.baselines import carver_rank, carver_support_reason
from experiments.common.kernels import make_case
from experiments.common.spec import Device, Workload
from experiments.xgboost.data import features
from experiments.xgboost.model import _libraries, _matrix
from experiments.xgboost.oracle_training import read_event_oracles


VERSION = 2
B200_LIMITS = {
    "sm_count": 148,
    "shared_memory_per_sm": 233472,
    "shared_memory_per_block": 232448,
    "registers_per_sm": 65536,
    "max_threads_per_sm": 2048,
    "max_threads_per_block": 1024,
    "warp_size": 32,
    "max_blocks_per_sm": 32,
}
TARGET = {"kind": "cuda", "arch": "sm_100a"}


def _read_json(path):
    return json.loads(Path(path).read_text())


def _exact_k(ranking, top_k):
    """Use ranking order and original-index tie breaking without tie expansion."""
    eligible = [row for row in ranking if row.get("tier") == "eligible"]
    return [row["index"] for row in eligible[:top_k]]


def _xgboost_selection(oracle, artifact, evaluation_indices, top_k, workers):
    np, xgb = _libraries()
    started = time.perf_counter()
    booster = xgb.Booster(params={"nthread": workers, "device": "cpu"})
    booster.load_model(bytearray(json.dumps(artifact["booster"]).encode()))
    rows = [features(oracle["context"], oracle["configs"][index]) for index in evaluation_indices]
    matrix = xgb.DMatrix(_matrix(rows, artifact["schema"]), nthread=workers)
    scores = booster.predict(matrix, iteration_range=(0, artifact["report"]["boosting"]["best_rounds"]))
    if len(scores) != len(evaluation_indices) or not np.isfinite(scores).all():
        raise ValueError(f"{oracle['name']}: invalid XGBoost predictions")
    order = sorted(
        range(len(evaluation_indices)),
        key=lambda position: (scores[position], evaluation_indices[position]),
    )
    selected = [evaluation_indices[position] for position in order[:top_k]]
    return selected, time.perf_counter() - started


def _carver_selection(oracle, evaluation_indices, top_k):
    workload = Workload(**oracle["experiment"]["workload"])
    device = Device("blackwell", TARGET, device_limits=B200_LIMITS)
    reason = carver_support_reason(workload, device)
    if reason:
        return None, 0.0, reason
    configs = [oracle["configs"][index] for index in evaluation_indices]
    started = time.perf_counter()
    report = carver_rank(workload, device, configs, top_k)
    selected = [evaluation_indices[index] for index in _exact_k(report["ranking"], top_k)]
    return selected, time.perf_counter() - started, None


def _tiletune_selection(oracle, evaluation_indices, top_k):
    from tilelang.tiletune import TileTuneConfig
    from tilelang.tiletune.runtime import TileTuneSession

    workload = Workload(**oracle["experiment"]["workload"])
    case = make_case(workload)
    configs = [oracle["configs"][index] for index in evaluation_indices]
    config = TileTuneConfig(
        enabled=True,
        mode="report_only",
        ranking_metric="memory",
        device_limits=B200_LIMITS,
        input_values=case.input_values or None,
    )
    session = TileTuneSession(config, configs, target=TARGET, device_limits=B200_LIMITS)
    started = time.perf_counter()
    for local_index, candidate in enumerate(configs):
        kwargs = {key: value for key, value in candidate.items() if key != "pass_configs"}
        pass_configs = {**(case.pass_configs or {}), **(candidate.get("pass_configs") or {})}
        try:
            session.elaborate(local_index, kwargs, case.build, pass_configs=pass_configs)
        except Exception:
            # The session retains the exact failure and the failed candidate is
            # not eligible for the shortlist.
            continue
    report = session.finish()
    selected = [evaluation_indices[index] for index in _exact_k(report["ranking"], top_k)]
    return selected, time.perf_counter() - started, dict(Counter(record["status"] for record in report["configs"]))


def _effect(oracle, evaluation_indices, selected, *, max_gap_percent):
    measured = {index: record["latency_ms"] for index, record in oracle["outcomes"].items() if record.get("status") == "benchmarked"}
    oracle_latencies = [measured[index] for index in evaluation_indices if index in measured]
    if not oracle_latencies:
        raise ValueError(f"{oracle['name']}: held-out pool has no successful oracle measurements")
    selected_latencies = [measured[index] for index in selected if index in measured]
    oracle_best = min(oracle_latencies)
    selected_best = min(selected_latencies) if selected_latencies else None
    gap = (selected_best / oracle_best - 1) * 100 if selected_best is not None else None
    return dict(
        evaluation_pool_size=len(evaluation_indices),
        evaluation_successful=len(oracle_latencies),
        selected_count=len(selected),
        selected_successful=len(selected_latencies),
        selected_indices=selected,
        heldout_oracle_latency_ms=oracle_best,
        selected_best_latency_ms=selected_best,
        latency_gap_percent=gap,
        oracle_at_k=oracle_best / selected_best if selected_best is not None else None,
        passes_under_50_percent=gap is not None and gap < max_gap_percent,
    )


def _cost(oracle, selected_count, selection_seconds, preparation):
    full_seconds = oracle["summary"]["tuning_seconds"]
    optional_validation_seconds = full_seconds * selected_count / len(oracle["configs"])
    preparation_gpu = preparation.get("label_collection_seconds", 0.0)
    preparation_cpu = preparation.get("cpu_training_seconds", 0.0)
    method_cost = preparation_gpu + preparation_cpu + selection_seconds
    return dict(
        source_exhaustive_one_gpu_seconds=full_seconds,
        preparation_gpu_estimated_seconds=preparation_gpu,
        preparation_cpu_measured_seconds=preparation_cpu,
        selection_cpu_measured_seconds=selection_seconds,
        method_cost_seconds=method_cost,
        method_cost_vs_exhaustive_percent=100 * method_cost / full_seconds,
        optional_shortlist_validation_gpu_estimated_seconds=optional_validation_seconds,
        end_to_end_with_optional_validation_seconds=method_cost + optional_validation_seconds,
        optional_gpu_estimate_method=("source one-GPU tuning_seconds multiplied by attempted configurations / full pool size"),
    )


def _method_summary(rows, total_workloads):
    available = [row for row in rows if row["status"] == "available"]
    passing = [row for row in available if row["effect"]["passes_under_50_percent"]]
    gaps = [row["effect"]["latency_gap_percent"] for row in available if row["effect"]["latency_gap_percent"] is not None]
    ratios = [row["effect"]["oracle_at_k"] for row in available if row["effect"]["oracle_at_k"] is not None]
    total_cost = sum(row["cost"]["method_cost_seconds"] for row in available)
    exhaustive = sum(row["cost"]["source_exhaustive_one_gpu_seconds"] for row in available)
    return dict(
        coverage=f"{len(available)}/{total_workloads}",
        available_workloads=len(available),
        unavailable_workloads=total_workloads - len(available),
        passing_workloads=len(passing),
        pass_rate_available_percent=100 * len(passing) / len(available) if available else None,
        pass_rate_all_workloads_percent=100 * len(passing) / total_workloads,
        mean_latency_gap_percent=statistics.fmean(gaps) if gaps else None,
        worst_latency_gap_percent=max(gaps) if gaps else None,
        mean_oracle_at_k=statistics.fmean(ratios) if ratios else None,
        exact_oracle_hits=sum(math.isclose(value, 1.0, rel_tol=1e-12) for value in ratios),
        total_selection_cpu_seconds=sum(row["cost"]["selection_cpu_measured_seconds"] for row in available),
        total_preparation_gpu_estimated_seconds=sum(row["cost"]["preparation_gpu_estimated_seconds"] for row in available),
        total_preparation_cpu_measured_seconds=sum(row["cost"]["preparation_cpu_measured_seconds"] for row in available),
        method_cost_seconds=total_cost,
        optional_shortlist_validation_gpu_estimated_seconds=sum(
            row["cost"]["optional_shortlist_validation_gpu_estimated_seconds"] for row in available
        ),
        source_exhaustive_seconds_for_covered_workloads=exhaustive,
        method_cost_vs_exhaustive_percent=100 * total_cost / exhaustive if exhaustive else None,
    )


def _format(value, digits=2):
    return "N/A" if value is None else f"{value:.{digits}f}"


def _write_markdown(path, report):
    top_k = report["protocol"]["top_k"]
    lines = [
        "# Carver, XGBoost, and TileTune effect/cost report",
        "",
        f"Generated: {report['recorded_at']}",
        "",
        (
            "Standard: best selected latency must be strictly below "
            f"{report['threshold']['max_latency_gap_percent']:.0f}% "
            f"over the held-out exhaustive oracle. Every method uses exact K={top_k} "
            "with original-index tie breaking."
        ),
        "",
        "## Summary",
        "",
        f"| Method | Workload coverage | Quality pass | Mean gap | Worst gap | Mean Oracle@{top_k} | "
        "Exact hits | Method cost | Cost/oracle |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("carver", "xgboost", "tiletune"):
        item = report["summary"][method]
        lines.append(
            f"| {method} | {item['coverage']} | {item['passing_workloads']}/{item['available_workloads']} | "
            f"{_format(item['mean_latency_gap_percent'])}% | {_format(item['worst_latency_gap_percent'])}% | "
            f"{_format(100 * item['mean_oracle_at_k'])}% | {item['exact_oracle_hits']} | "
            f"{_format(item['method_cost_seconds'])} s | "
            f"{_format(item['method_cost_vs_exhaustive_percent'])}% |"
        )
    lines += [
        "",
        "## Per-workload results",
        "",
        "| Workload | Method | Status | K/success | Oracle ms | Selected ms | Gap | Pass | "
        "Select CPU s | Train CPU s | Train/val GPU est. s | Method cost s | Optional K GPU est. s |",
        "|---|---|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        if row["status"] != "available":
            lines.append(f"| {row['workload']} | {row['method']} | unavailable: {row['reason']} | — | — | — | — | no | — | — | — | — | — |")
            continue
        effect, cost = row["effect"], row["cost"]
        lines.append(
            f"| {row['workload']} | {row['method']} | available | "
            f"{effect['selected_count']}/{effect['selected_successful']} | "
            f"{_format(effect['heldout_oracle_latency_ms'], 6)} | {_format(effect['selected_best_latency_ms'], 6)} | "
            f"{_format(effect['latency_gap_percent'])}% | {'yes' if effect['passes_under_50_percent'] else 'no'} | "
            f"{_format(cost['selection_cpu_measured_seconds'], 3)} | "
            f"{_format(cost['preparation_cpu_measured_seconds'], 3)} | "
            f"{_format(cost['preparation_gpu_estimated_seconds'], 3)} | "
            f"{_format(cost['method_cost_seconds'], 3)} | "
            f"{_format(cost['optional_shortlist_validation_gpu_estimated_seconds'], 3)} |"
        )
    lines += [
        "",
        "## Accounting notes",
        "",
        (
            "- The evaluation pool is the frozen 80% held-out configuration split from the per-workload "
            "XGBoost study; all methods see the same pool."
        ),
        (
            "- Workload coverage means the method can produce a ranking for that workload. Quality pass "
            "means the selected best latency is strictly less than 1.5x the held-out oracle best."
        ),
        ("- XGBoost cost includes its estimated 10% training plus 10% validation GPU label collection and measured CPU training pipeline."),
        (
            "- Carver and TileTune require no latency-label training. TileTune uses the B200 memory "
            "ranking used by the final system protocol, so its method cost is CPU-only."
        ),
        (
            f"- The optional K={top_k} GPU validation estimate is shown separately and is not charged to "
            "method cost. No kernels were rerun; effects use the already-collected oracle."
        ),
        "- Carver unsupported workloads are reported as unavailable and are not assigned a fallback model.",
        ("- Aggregate cost/oracle uses only the workloads covered by that method."),
        "",
    ]
    path.write_text("\n".join(lines))


def generate_report(oracle_root, model_root, output, *, top_k=20, max_gap_percent=50.0, workers=8):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    model_root = Path(model_root).resolve()
    oracles = read_event_oracles(oracle_root)
    results = []
    for oracle in oracles:
        artifact_path = model_root / f"{oracle['name']}.json"
        artifact = _read_json(artifact_path)
        model_report = artifact["report"]
        if model_report["provenance"]["outcomes_sha256"] != oracle["provenance"]["outcomes_sha256"]:
            raise ValueError(f"{oracle['name']}: XGBoost model and oracle provenance differ")
        evaluation_indices = model_report["split"]["heldout_indices"]
        preparations = {
            "xgboost": dict(
                label_collection_seconds=model_report["label_collection_estimate"]["total_collection_seconds"],
                cpu_training_seconds=model_report["cpu_timing"]["training_pipeline_seconds"],
            )
        }

        selected, seconds, reason = _carver_selection(oracle, evaluation_indices, top_k)
        if reason:
            results.append(dict(workload=oracle["name"], method="carver", status="unavailable", reason=reason))
        else:
            effect = _effect(oracle, evaluation_indices, selected, max_gap_percent=max_gap_percent)
            results.append(
                dict(
                    workload=oracle["name"],
                    method="carver",
                    status="available",
                    effect=effect,
                    cost=_cost(oracle, len(selected), seconds, {}),
                )
            )

        selected, seconds = _xgboost_selection(oracle, artifact, evaluation_indices, top_k, workers)
        effect = _effect(oracle, evaluation_indices, selected, max_gap_percent=max_gap_percent)
        results.append(
            dict(
                workload=oracle["name"],
                method="xgboost",
                status="available",
                effect=effect,
                cost=_cost(oracle, len(selected), seconds, preparations["xgboost"]),
            )
        )

        selected, seconds, statuses = _tiletune_selection(oracle, evaluation_indices, top_k)
        effect = _effect(oracle, evaluation_indices, selected, max_gap_percent=max_gap_percent)
        results.append(
            dict(
                workload=oracle["name"],
                method="tiletune",
                status="available",
                analysis_statuses=statuses,
                effect=effect,
                cost=_cost(oracle, len(selected), seconds, {}),
            )
        )
        print(
            f"{oracle['name']}: "
            + ", ".join(
                (
                    f"{row['method']}=N/A"
                    if row["status"] != "available"
                    else f"{row['method']}={_format(row['effect']['latency_gap_percent'])}%"
                )
                for row in results[-3:]
            ),
            flush=True,
        )

    by_method = {method: [row for row in results if row["method"] == method] for method in ("carver", "xgboost", "tiletune")}
    report = dict(
        version=VERSION,
        recorded_at=datetime.now(timezone.utc).isoformat(),
        oracle_root=str(Path(oracle_root).resolve()),
        model_root=str(model_root),
        protocol=dict(
            top_k=top_k,
            tie_break="original configuration index",
            evaluation_pool="frozen XGBoost held-out split, shared by all methods",
            xgboost_ranking="log-latency prediction",
            carver_ranking="family Carver adapter",
            tiletune_ranking="B200 memory metric",
            kernel_execution="none; effects use exhaustive E1 outcomes",
        ),
        threshold=dict(
            max_latency_gap_percent=max_gap_percent,
            rule="selected_best_latency / heldout_oracle_best_latency - 1 < threshold",
        ),
        summary={method: _method_summary(rows, len(oracles)) for method, rows in by_method.items()},
        results=results,
    )
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    _write_markdown(output / "report.md", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-gap-percent", type=float, default=50.0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.top_k <= 0 or args.max_gap_percent <= 0 or args.workers <= 0:
        parser.error("top-k, max gap, and workers must be positive")
    report = generate_report(
        args.oracle_root,
        args.model_root,
        args.output,
        top_k=args.top_k,
        max_gap_percent=args.max_gap_percent,
        workers=args.workers,
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
