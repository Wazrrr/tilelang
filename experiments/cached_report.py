"""Render the cache-first study's immutable measurements without a GPU."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from experiments.utils.io import write_json
from experiments.utils.results import read


def geomean(values):
    return math.exp(statistics.mean(map(math.log, values))) if values else None


def number(value, digits=3):
    return "N/A" if value is None else f"{value:.{digits}f}"


def report(root):
    root = Path(root).resolve()
    plan = read(root / "plan.json")
    rows, cases, models, seen = [], [], {}, set()
    for path in sorted((root / "comparison").glob("*/*/comparison.json")):
        case = read(path)
        curves = read(path.parent / "oracle-curves.json")
        cases.append(dict(**case, curves=curves))
        for method, result in case["methods"].items():
            # Fixed baselines are charged once per shape, independently of repeats.
            key = (case["workload"]["name"], method, case["seed"] if method == "tiletune" else None)
            if key in seen:
                continue
            seen.add(key)
            curve = next((c for c in curves["methods"] if c["method"] == method), None)
            prefix = next((c for c in curve["curves"] if c["k"] == 20), {}) if curve else {}
            if method == "brute_force":
                prefix = dict(
                    best_latency_ms=curves["oracle"]["best_latency_ms"],
                    oracle_at_k=1,
                    best_config=curves["oracle"]["best_config"],
                    successful_count=curves["oracle"]["statuses"]["benchmarked"],
                )
            # Ranking prefixes and actual frozen selections are both preserved.
            selected = (curve or {}).get("saved_selection") or {}
            rows.append(
                dict(
                    family=case["family"],
                    workload=key[0],
                    seed=key[2],
                    method=method,
                    dtype=case["workload"]["dtype"],
                    parameters=json.dumps(case["workload"]["parameters"], sort_keys=True),
                    status=result["status"],
                    reason=result.get("reason"),
                    oracle_ms=curves["oracle"]["best_latency_ms"],
                    top20_best_ms=prefix.get("best_latency_ms"),
                    oracle_at_20=prefix.get("oracle_at_k"),
                    selected_best_ms=selected.get("best_latency_ms"),
                    selected_oracle_at_k=selected.get("oracle_at_k"),
                    measured_winner_ms=(result.get("winner") or {}).get("latency_ms"),
                    online_tuning_seconds=result.get("tuning_seconds"),
                    worker_seconds=result.get("worker_wall_seconds", result.get("worker_seconds")),
                    timing_scope=result.get("timing_scope", "selection + compilation + benchmarking"),
                    selected_count=(result.get("selection") or {}).get("selected_count"),
                    top20_successful_count=prefix.get("successful_count"),
                    measured_best_config=json.dumps((result.get("winner") or {}).get("config"), sort_keys=True),
                    top20_best_config=json.dumps(prefix.get("best_config"), sort_keys=True),
                    record_path=str((path.parent / method).resolve()),
                )
            )
        model_ref = (
            Path(case["baseline_bundle"])
            / "collection"
            / plan["devices"][0]["name"]
            / "test"
            / case["workload"]["name"]
            / "xgboost-model.json"
        )
        if model_ref.exists():
            ref = read(model_ref)
            model = Path(ref["path"])
            checksum = hashlib.sha256(model.read_bytes()).hexdigest()
            if checksum != ref["sha256"]:
                raise ValueError(f"XGBoost model changed: {model}")
            artifact = read(model)
            models[case["family"]] = dict(path=str(model), sha256=checksum, training=artifact["training"], parameters=artifact["params"])
    aggregate = {}
    for method in ("brute_force", "carver", "xgboost", "tiletune"):
        subset = [r for r in rows if r["method"] == method]
        valid = [r["oracle_at_20"] for r in subset if r["oracle_at_20"] is not None]
        times = [r["online_tuning_seconds"] for r in subset if r["online_tuning_seconds"] is not None]
        aggregate[method] = dict(
            records=len(subset),
            valid_records=len(valid),
            geometric_mean_oracle_at_20=geomean(valid),
            timing_records=len(times),
            median_online_seconds=statistics.median(times) if times else None,
            total_online_seconds=sum(times) if len(times) == len(subset) else None,
        )
    expected = len(plan["splits"]["test"]) * len(plan["budget"]["seeds"])
    summary = dict(
        status="completed" if len(cases) == expected else "incomplete",
        expected_comparisons=expected,
        completed_comparisons=len(cases),
        aggregate=aggregate,
        rows=rows,
        cases=cases,
        xgboost_models=models,
        baselines=read(root / "baselines.json"),
        profiles=read(root / "profiles.json"),
        semantics=dict(
            oracle_at_20="oracle minimum / minimum successful oracle latency in first 20 ranked configurations",
            cost="Original measured selection, compilation and benchmarking cost; cache lookup is not new tuning time",
            failure_policy="Compilation and correctness failures consume the fixed budget; no replacements",
            contention="Reject any invocation with observed foreign compute processes; retry on an idle GPU",
            polling_interval_seconds=1,
            polling_limit="Subsecond overlap cannot be ruled out by process polling",
        ),
    )
    write_json(root / "comparison.json", summary)
    with (root / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["workload"])
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Cached kernel study",
        "",
        f"Status: **{summary['status']}**; {len(cases)}/{expected} shape/seed comparisons.",
        "",
        "Each family has five final shapes. Baselines are fixed; TileTune repeats reuse them. All reported shortlist latencies come from the same exhaustive table. Oracle@20 is oracle latency divided by the best successful latency in the first 20 ranked candidates; 100% is optimal. Failed candidates consume budget without replacement.",
        "",
        "## Aggregate comparison",
        "",
        "| Method | Valid / total records | Geomean Oracle@20 | Median online seconds |",
        "|---|---:|---:|---:|",
    ]
    for method, item in aggregate.items():
        ratio = item["geometric_mean_oracle_at_20"]
        lines.append(
            f"| {method} | {item['valid_records']} / {item['records']} | {number(100 * ratio if ratio is not None else None, 2)}% | {number(item['median_online_seconds'], 2)} |"
        )
    lines += [
        "",
        "Aggregates exclude unavailable measurements and show coverage. Fixed baselines are counted once per shape; TileTune has one record per requested seed. Brute-force cost is the sum of accepted shard work across GPUs, not parallel elapsed time.",
        "",
        "## Per-shape quality",
        "",
        "TileTune values are medians across repeats. Full per-seed values, actual measured winners, configuration dictionaries and cost breakdowns are in `comparison.csv` and `comparison.json`.",
        "",
        "| Shape | Oracle ms | Carver ms / Oracle@20 | XGBoost ms / Oracle@20 | TileTune ms / Oracle@20 |",
        "|---|---:|---:|---:|---:|",
    ]
    for workload in plan["splits"]["test"]:
        subset = [r for r in rows if r["workload"] == workload["name"]]
        cells = [workload["name"], number(subset[0]["oracle_ms"], 6) if subset else "N/A"]
        for method in ("carver", "xgboost", "tiletune"):
            values = [r for r in subset if r["method"] == method and r["oracle_at_20"] is not None]
            cells.append(
                number(statistics.median(r["top20_best_ms"] for r in values), 6)
                + " / "
                + number(100 * statistics.median(r["oracle_at_20"] for r in values), 2)
                + "%"
                if values
                else "N/A"
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Per-shape online tuning cost",
        "",
        "| Shape | Brute-force aggregate s | Carver s | XGBoost s | TileTune median s |",
        "|---|---:|---:|---:|---:|",
    ]
    for workload in plan["splits"]["test"]:
        cells = [workload["name"]]
        for method in ("brute_force", "carver", "xgboost", "tiletune"):
            values = [
                r["online_tuning_seconds"]
                for r in rows
                if r["workload"] == workload["name"] and r["method"] == method and r["online_tuning_seconds"] is not None
            ]
            cells.append(number(statistics.median(values) if values else None, 2))
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Reusable XGBoost preparation",
        "",
        "Two training shapes and one validation shape per family; 10% of each pool. Test shapes are held out. Early stopping uses validation data only.",
        "",
        "| Family | Training / validation samples | Data collection s | Fit s | Validation log RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for family, model in models.items():
        t = model["training"]
        collection = [t.get(k) for k in ("training_collection_seconds", "validation_collection_seconds")]
        lines.append(
            f"| {family} | {t['training_samples']} / {t['validation_samples']} | {number(sum(collection) if all(v is not None for v in collection) else None, 2)} | {number(t['fit_seconds'], 2)} | {number(t['validation_log_rmse'], 4)} |"
        )
    lines += [
        "",
        "Training/validation measurements, models, oracle outcomes, method rankings, best configurations and their hashes remain in immutable family baseline bundles. TileTune records are cached by model-code revision, pool, shape, profile, seed and budget. Primitive profiles and winner validation samples are also retained.",
        "",
        "## Measurement validity",
        "",
        "Each accepted worker has a finalized contention audit. Foreign GPU processes cause measurements to be discarded and retried. Process polling is once per second, so overlap shorter than that interval cannot be excluded. Native results apply only to the observed GPU model. No model accuracy is inferred from failed or unavailable selections.",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    report(parser.parse_args().output)
