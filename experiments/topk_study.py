"""Sweep saved rankings against a completed study's exhaustive oracle tables.

python -m experiments.topk_study --study experiments/results/studies/STUDY

This is a retrospective, standard-library-only analysis. It does not retrain,
rerank, benchmark, or estimate online tuning time at the larger budgets.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from experiments.compare_results import compare
from experiments.utils.results import provenance, read


def budget_levels(pool_size, ks, percents):
    """Percentage budgets round up and use the entire declared pool."""
    if any(type(k) is not int or k <= 0 for k in ks):
        raise ValueError("top-K values must be positive integers")
    if any(type(p) is not int or not 1 <= p <= 100 for p in percents):
        raise ValueError("pool percentages must be integers in [1, 100]")
    return {**{f"K={k}": k for k in ks}, **{f"{p}%": (pool_size * p + 99) // 100 for p in percents}}


def aggregate(rows):
    """All-hit cutoffs require every case/seed, never a median across cases."""
    hits = [r for r in rows if r["first_oracle_hit_k"] is not None]
    complete = len(hits) == len(rows) and bool(rows)
    return dict(
        records=len(rows),
        reachable_records=len(hits),
        all_hit_k=max(r["first_oracle_hit_k"] for r in hits) if complete else None,
        max_hit_rank_pool_percent=max(100 * r["first_oracle_hit_k"] / r["pool_size"] for r in hits) if complete else None,
        all_hit_whole_pool_percent=(
            next(p for p in range(1, 101) if all((r["pool_size"] * p + 99) // 100 >= r["first_oracle_hit_k"] for r in rows))
            if complete
            else None
        ),
        unreachable=[dict(workload=r["workload"], seed=r["seed"]) for r in rows if r["first_oracle_hit_k"] is None],
    )


def summarize_budgets(rows):
    summaries = []
    for label in rows[0]["budgets"]:
        values = [r["budgets"][label] for r in rows]
        ratios = [v["oracle_at_k"] for v in values if v["oracle_at_k"] is not None]
        summaries.append(
            dict(
                budget=label,
                exact_hits=sum(v["exact_hit"] for v in values),
                records=len(rows),
                valid_records=len(ratios),
                geometric_mean_oracle_at_k=math.exp(statistics.mean(map(math.log, ratios))) if ratios else None,
                worst_oracle_at_k=min(ratios) if ratios else None,
                shortfall_records=sum(v["shortfall"] > 0 for v in values),
            )
        )
    return summaries


def sweep(study, ks, percents, methods):
    study = Path(study).resolve()
    source = read(study / "comparison.json")
    expected = source.get("expected_comparisons", source.get("expected_shapes"))
    paths = sorted((study / "comparison").glob("*/*/oracle-curves.json"))
    if source.get("status") != "completed" or not expected or len(paths) != expected:
        raise ValueError("a completed study with every case's oracle-curves.json is required")
    rows, evidence, seen = [], [], {}
    for path in paths:
        case = read(path.parent / "comparison.json")
        saved = read(path)
        workload = case["workload"]
        seed = case.get("seed", source.get("protocol", {}).get("seed"))
        family = case.get("family", path.parent.parent.name)
        levels = budget_levels(saved["oracle"]["candidate_count"], ks, percents)
        # Always include the full pool to establish each method's ceiling.
        levels["100%"] = saved["oracle"]["candidate_count"]
        references = {m["method"]: m for m in saved["methods"]}
        chosen = {}
        for method in methods:
            ref = references[method]["input"]
            key = (workload["name"], method, seed if method == "tiletune" else None)
            if key in seen:
                if seen[key] != ref["sha256"]:
                    raise ValueError(f"repeated baseline/ranking changed: {key}")
                continue
            seen[key] = ref["sha256"]
            chosen[method] = ref["path"]
        if not chosen:
            continue
        result = compare(saved["oracle"]["sources"][0]["path"], chosen, sorted(set(levels.values()) | {20}))
        if result["oracle"]["sources"] != saved["oracle"]["sources"]:
            raise ValueError(f"oracle changed since the original comparison: {path}")
        if result["oracle"]["candidate_count"] != saved["oracle"]["candidate_count"]:
            raise ValueError(f"oracle pool changed: {path}")
        for method in result["methods"]:
            name = method["method"]
            if method["input"] != references[name]["input"]:
                raise ValueError(f"ranking changed since the original comparison: {path}, {name}")
            curves = {c["k"]: c for c in method["curves"]}
            old20 = next((c for c in references[name]["curves"] if c["k"] == 20), None)
            if saved.get("version") == result["version"] and old20 is not None and curves[20]["oracle_at_k"] != old20["oracle_at_k"]:
                raise ValueError(f"Oracle@20 differs from the original comparison: {path}, {name}")
            if method["status"] == "evaluated" and method["candidate_count"] != result["oracle"]["candidate_count"]:
                raise ValueError(f"method does not record the full configuration pool: {path}, {name}")
            first = method.get("first_oracle_hit_k")
            budgets = {}
            for label, k in levels.items():
                curve = curves.get(k, {})
                budgets[label] = dict(
                    k=k,
                    selected_count=curve.get("selected_count", 0),
                    successful_count=curve.get("successful_count", 0),
                    shortfall=curve.get("shortfall", k),
                    oracle_at_k=curve.get("oracle_at_k"),
                    best_latency_ms=curve.get("best_latency_ms"),
                    exact_hit=first is not None and first <= k,
                )
            rows.append(
                dict(
                    family=family,
                    workload=workload["name"],
                    dtype=workload["dtype"],
                    parameters=workload["parameters"],
                    seed=seed if name == "tiletune" else None,
                    method=name,
                    status=method["status"],
                    pool_size=result["oracle"]["candidate_count"],
                    eligible_count=method.get("available_count", 0),
                    first_oracle_hit_k=first,
                    hit_rank_pool_percent=100 * first / result["oracle"]["candidate_count"] if first is not None else None,
                    oracle_ms=result["oracle"]["best_latency_ms"],
                    oracle_candidates=method.get("oracle_candidates", []),
                    budgets=budgets,
                )
            )
        evidence.append(dict(comparison=provenance(path), workload=workload, seed=seed, result=result))
        print(f"Evaluated {workload['name']} seed={seed}", flush=True)
    groups = {}
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        groups[method] = dict(
            **aggregate(subset),
            budgets=summarize_budgets(subset),
            families={family: aggregate([r for r in subset if r["family"] == family]) for family in sorted({r["family"] for r in subset})},
        )
    return dict(
        version=2,
        study=provenance(study / "comparison.json"),
        completed_comparisons=len(paths),
        semantics=dict(
            scope="Retrospective evaluation of the frozen study rankings and oracle measurements; no new GPU timing or model training",
            exact_hit="At least one candidate has exactly the minimum recorded oracle latency; tied minima count, rounded 100% does not",
            percentages="ceil(percent * total declared pool / 100), including failed and model-excluded configurations in the denominator",
            order="Finite eligible scores in saved order; unknown/rejected candidates remain excluded; compile/check failures consume K",
            cutoff="Maximum first-hit primary-score group tail rank across every case/seed; null when any oracle is unreachable",
            ties="Only complete equal-primary-score groups within K count in ranking curves; saved selections retain their actual contents",
            repeats="TileTune includes every saved seed; fixed baselines count once per workload",
        ),
        aggregate=groups,
        rows=rows,
        evidence=evidence,
    )


def render(result):
    lines = [
        "# Oracle hits at larger top-K budgets",
        "",
        f"Source: `{Path(result['study']['path']).parent.name}`; {result['completed_comparisons']} completed comparisons.",
        "",
        "This reuses the frozen rankings and exhaustive measurements of that study. It does not establish results for newer kernel/model revisions or measure larger-budget tuning time.",
        "",
        "An exact hit requires the saved minimum latency (any tied optimum counts). Percentages use the entire declared pool and round K up. Failed candidates consume budget; unknown/rejected candidates stay excluded. Shortfalls are reported even when a prefix contains an optimum.",
        "",
        "## All-case cutoffs",
        "",
        "| Method | Reachable / total | K for all hits | Whole-pool percentage for all hits |",
        "|---|---:|---:|---:|",
    ]
    for name, group in result["aggregate"].items():
        k, p = group["all_hit_k"], group["all_hit_whole_pool_percent"]
        lines.append(
            f"| {name} | {group['reachable_records']}/{group['records']} | {k if k is not None else 'Unreachable'} | {str(p) + '%' if p is not None else 'Unreachable'} |"
        )
    lines += [
        "",
        "## Budget sweep",
        "",
        "| Method | Budget | Exact hits | Geomean Oracle@K | Worst Oracle@K | Shortfalls |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, group in result["aggregate"].items():
        for b in group["budgets"]:
            gm, worst = b["geometric_mean_oracle_at_k"], b["worst_oracle_at_k"]
            gm = f"{100 * gm:.3f}%" if gm is not None else "N/A"
            worst = f"{100 * worst:.3f}%" if worst is not None else "N/A"
            lines.append(f"| {name} | {b['budget']} | {b['exact_hits']}/{b['records']} | {gm} | {worst} | {b['shortfall_records']} |")
    lines += [
        "",
        "## Cutoffs by family",
        "",
        "| Family | Method | Reachable / total | K for all hits | Whole-pool percentage |",
        "|---|---|---:|---:|---:|",
    ]
    for name, group in result["aggregate"].items():
        for family, values in group["families"].items():
            k, p = values["all_hit_k"], values["all_hit_whole_pool_percent"]
            lines.append(
                f"| {family} | {name} | {values['reachable_records']}/{values['records']} | {k if k is not None else 'Unreachable'} | {str(p) + '%' if p is not None else 'Unreachable'} |"
            )
    lines += [
        "",
        "## Per-case oracle positions",
        "",
        "`Unreachable` means the full eligible ranking excludes every exact oracle optimum.",
        "",
        "| Case | Method | Seed | Pool | Eligible | First-hit K | Rank / pool | Full eligible Oracle@K |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        first, share, ceiling = row["first_oracle_hit_k"], row["hit_rank_pool_percent"], row["budgets"]["100%"]["oracle_at_k"]
        share = f"{share:.3f}%" if share is not None else "—"
        ceiling = f"{100 * ceiling:.3f}%" if ceiling is not None else "N/A"
        lines.append(
            f"| {row['workload']} | {row['method']} | {row['seed'] if row['seed'] is not None else 'fixed'} | {row['pool_size']} | {row['eligible_count']} | {first if first is not None else 'Unreachable'} | {share} | {ceiling} |"
        )
    lines += ["", "## Excluded oracle diagnostics", ""]
    for row in result["rows"]:
        if row["first_oracle_hit_k"] is not None:
            continue
        lines.append(f"### {row['workload']} / {row['method']} / seed {row['seed']}")
        lines.append("")
        for candidate in row["oracle_candidates"]:
            lines.append(
                f"- Oracle index {candidate['oracle_index']}, tier `{candidate['tier']}`: `{json.dumps(candidate['config'], sort_keys=True)}`"
            )
            reasons = [d["reason"] for d in candidate["diagnostics"]] + candidate["model_unknown"]
            for reason in dict.fromkeys(reasons):
                lines.append(f"- {reason}")
        lines.append("")
    lines += [
        "Full per-budget case data is in `topk.csv`; `topk.json` retains source hashes, ordered prefixes, winners and diagnostics.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="default: STUDY/topk-study")
    parser.add_argument("--top-k", type=int, nargs="+", default=[20, 50, 100, 200, 500, 1000])
    parser.add_argument("--pool-percent", type=int, nargs="+", default=[5, 10, 20, 25, 50, 75, 100])
    parser.add_argument("--methods", nargs="+", choices=("tiletune", "xgboost", "carver"), default=["tiletune", "xgboost"])
    args = parser.parse_args(argv)
    output = args.output or args.study / "topk-study"
    if output.resolve() == args.study.resolve():
        parser.error("use a separate output directory to preserve the original study report")
    result = sweep(args.study, args.top_k, args.pool_percent, args.methods)
    output.mkdir(parents=True, exist_ok=True)
    (output / "topk.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output / "report.md").write_text(render(result))
    csv_rows = []
    for row in result["rows"]:
        base = {k: v for k, v in row.items() if k not in ("budgets", "oracle_candidates", "parameters")}
        csv_rows.extend(dict(base, budget=label, **values) for label, values in row["budgets"].items())
    with (output / "topk.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Completed {output / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
