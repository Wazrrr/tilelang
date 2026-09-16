"""Compare saved top-K configurations with a common measured oracle, offline.

Only the Python standard library is needed. Configuration dictionaries are the
join key: local/original indices may differ across saved reports. All latencies
come from the oracle table, never from model scores or separate winner reruns.
"""

import argparse
import json
from pathlib import Path

from experiments.utils.results import (
    UNAVAILABLE,
    check_metadata,
    config_key,
    finite,
    load_oracle,
    measure_prefix,
    provenance,
    read,
    records_by_index,
    validate_order,
)


def compare_method(name, path, oracle, ks, order):
    path = Path(path).resolve()
    if path.is_dir():
        path = path / (f"{name}.json" if (path / f"{name}.json").is_file() else "result.json")
    report = read(path)
    result = dict(method=name, input=provenance(path))
    if report.get("status") in UNAVAILABLE and not isinstance(report.get("configs"), list):
        return dict(result, status=report["status"], reason=report.get("reason"), curves=[], saved_selection=None)
    records = records_by_index(report.get("configs"))
    for row in records.values():
        if config_key(row["config"]) not in oracle["records"]:
            raise ValueError(f"{name} config at index {row['index']} is missing from the oracle pool")
    result["metadata_check"] = check_metadata(oracle["path"], path)
    selection = report.get("selection")
    selected = validate_order(selection["selected_indices"], records) if selection is not None else None
    if order == "ranking":
        ranking = report.get("ranking")
        if not isinstance(ranking, list):
            raise ValueError(f"{name}: no saved ranking; use --order selected for an ordered selection-only report")
        validate_order([r["index"] for r in ranking], records)
        if len(ranking) != len(records):
            raise ValueError(f"{name}: incomplete ranking; use --order selected for saved selections only")
        # Preserve saved tie order, excluding only the model's own unscored or
        # rejected entries. A compile/check failure still consumes its slot.
        indices = [r["index"] for r in ranking if r.get("tier") == "eligible" and finite(r.get("score"))]
    else:
        if selected is None:
            raise ValueError(f"{name}: no saved selected_indices")
        indices = selected
    budget = selection.get("requested_k", len(selected)) if selection is not None else None
    if selection is not None and (type(budget) is not int or budget <= 0 or budget < len(selected)):
        raise ValueError(f"{name}: invalid saved selection budget")
    return dict(
        result,
        status="evaluated",
        available_count=len(indices),
        curves=[measure_prefix(indices, records, oracle, k, order) for k in ks],
        saved_selection=measure_prefix(selected, records, oracle, budget, "saved") if selected is not None else None,
    )


def compare(oracle_path, methods, ks, order="ranking"):
    if not ks or any(type(k) is not int or k <= 0 for k in ks):
        raise ValueError("top-K values must be positive integers")
    oracle = load_oracle(oracle_path)
    winner = oracle["winner"]
    return dict(
        version=1,
        semantics=dict(
            oracle_at_k="oracle best latency / fastest successful oracle latency in the first K configs",
            timing="all latencies from the same correctness-checked oracle table, in milliseconds",
            order="saved ranking order (finite eligible scores) or explicitly requested saved selection order",
            failures="consume K; no replacement; no successful config yields null, not zero latency",
            shortfall="fewer than K available configs; percentage describes only the reported selected_count",
            curves="retrospective prefix evaluation; saved_selection separately evaluates the actual recorded shortlist",
        ),
        oracle=dict(
            sources=oracle["sources"],
            candidate_count=len(oracle["records"]),
            statuses=oracle["statuses"],
            best_index=winner["index"],
            best_config=winner["config"],
            best_latency_ms=winner["latency_ms"],
        ),
        methods=[compare_method(name, path, oracle, sorted(set(ks)), order) for name, path in methods.items()],
    )


def render(result):
    oracle = result["oracle"]
    lines = [
        f"Oracle best: {oracle['best_latency_ms']:.8f} ms; index {oracle['best_index']}; "
        f"{oracle['statuses']['benchmarked']}/{oracle['candidate_count']} valid configs",
        "Oracle config: " + json.dumps(oracle["best_config"], sort_keys=True),
        "Oracle@K = oracle best latency / best latency in the first K configs (100% is optimal).",
        "All timings use the oracle table. 'saved' is the actual recorded shortlist; other rows are prefix evaluations.",
        "",
        "| Method | Source | K | Configs | Valid | Best ms | Oracle@K | Latency gap | Best oracle index |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in result["methods"]:
        if method["status"] != "evaluated":
            lines.append(f"| {method['method']} | {method['status']} | — | — | — | — | — | — | — |")
            continue
        rows = method["curves"] + ([method["saved_selection"]] if method["saved_selection"] is not None else [])
        for row in rows:
            ms = f"{row['best_latency_ms']:.8f}" if row["best_latency_ms"] is not None else "N/A"
            pct = f"{row['oracle_percent']:.2f}%" if row["oracle_percent"] is not None else "N/A"
            gap = f"+{row['latency_gap_percent']:.2f}%" if row["latency_gap_percent"] is not None else "N/A"
            count = str(row["selected_count"]) + (" (shortfall)" if row["shortfall"] else "")
            index = row["best_oracle_index"] if row["best_oracle_index"] is not None else "—"
            lines.append(
                f"| {method['method']} | {row['source']} | {row['k']} | {count} | {row['successful_count']} | "
                f"{ms} | {pct} | {gap} | {index} |"
            )
    for method in result["methods"]:
        if method.get("reason"):
            lines.append(f"\n{method['method']}: {method['reason']}")
        if "metadata_check" in method:
            lines.append(f"\n{method['method']} metadata: {method['metadata_check']}.")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True, help="oracle.json, outcomes.json, brute_force.json or heuristic reference")
    for method in ("tiletune", "carver", "xgboost"):
        parser.add_argument(f"--{method}", type=Path, help=f"saved {method}.json or its containing directory")
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 5, 10, 20, 50])
    parser.add_argument("--order", choices=("ranking", "selected"), default="ranking")
    parser.add_argument("--output", type=Path, help="optional JSON report, including selected indices and winning configs")
    args = parser.parse_args(argv)
    methods = {name: getattr(args, name) for name in ("tiletune", "carver", "xgboost") if getattr(args, name) is not None}
    if not methods:
        parser.error("supply at least one of --tiletune, --carver or --xgboost")
    try:
        result = compare(args.oracle, methods, args.top_k, args.order)
        if args.output:
            inputs = [s["path"] for s in result["oracle"]["sources"]] + [m["input"]["path"] for m in result["methods"]]
            if str(args.output.resolve()) in inputs:
                raise ValueError("output must not overwrite an input result")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
