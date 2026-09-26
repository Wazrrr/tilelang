"""Extend the saved H200 cost-model report with normalized bound-aware rankings.

The input rankings and exhaustive measurements are immutable artifacts.  This
script only joins them by configuration identity, computes strict top-50%
statistics, and writes a new report.  A workload succeeds when either its E1
or its E2 exact oracle has conservative tail rank at most floor(pool_size / 2).
"""

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

from experiments.compare_results import compare
from experiments.utils.results import provenance


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIOR = ROOT / "experiments/results/h200-cost-model-comparison-25-20260924"
DEFAULT_BOUND_AWARE = ROOT / "experiments/results/bound-aware-normalized-validation-20260926"
DEFAULT_BASELINE = ROOT / "experiments/results/h200-cost-model-comparison-bound-aware-h200-ridge-25-20260925"
DEFAULT_OUTPUT = ROOT / "experiments/results/h200-cost-model-comparison-bound-aware-normalized-25-20260926"
METHODS = ("tiletune_memory", "bound_aware", "xgboost", "carver")
LABELS = {
    "tiletune_memory": "TileTune memory",
    "bound_aware": "TileTune bound-aware",
    "xgboost": "XGBoost",
    "carver": "Carver",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def staged_provenance(staged_path, final_path):
    return {
        "path": str(Path(final_path).resolve()),
        "sha256": hashlib.sha256(Path(staged_path).read_bytes()).hexdigest(),
    }


def oracle_pass(oracle, budget):
    rank = oracle.get("first_oracle_hit_k")
    return rank is not None and rank <= budget


def derive_method(method, budget, pool_size):
    method = copy.deepcopy(method)
    if not method.get("supported"):
        method.update(
            oracle_reachable_either=False,
            oracle_reachable_both=False,
            minimum_oracle_tail_rank=None,
            maximum_oracle_tail_rank=None,
            minimum_oracle_tail_fraction=None,
            maximum_oracle_tail_fraction=None,
            within_50_percent_either_oracle=False,
            within_50_percent_both_oracles=False,
            successful=False,
            passing_oracles=[],
        )
        return method
    ranks = [oracle.get("first_oracle_hit_k") for oracle in method["oracles"].values()]
    reachable = [rank for rank in ranks if rank is not None]
    passing = [name for name, oracle in method["oracles"].items() if oracle_pass(oracle, budget)]
    method.update(
        oracle_reachable_either=bool(reachable),
        oracle_reachable_both=len(reachable) == len(method["oracles"]),
        minimum_oracle_tail_rank=min(reachable) if reachable else None,
        maximum_oracle_tail_rank=max(reachable) if reachable else None,
        minimum_oracle_tail_fraction=min(reachable) / pool_size if reachable else None,
        maximum_oracle_tail_fraction=max(reachable) / pool_size if reachable else None,
        within_50_percent_either_oracle=bool(passing),
        within_50_percent_both_oracles=len(passing) == len(method["oracles"]),
        successful=bool(passing),
        passing_oracles=passing,
    )
    return method


def comparison_oracle(result, comparison_path):
    method = result["methods"][0]
    curve = method["curves"][0]
    return {
        "status": method["status"],
        "comparison": comparison_path,
        "input": method["input"],
        "first_oracle_hit_k": method["first_oracle_hit_k"],
        "oracle_reachable": method["first_oracle_hit_k"] is not None,
        "oracle_at_50": curve["oracle_at_k"],
        "oracle_percent_at_50": curve["oracle_percent"],
        "selected_count_at_50": curve["selected_count"],
        "successful_count_at_50": curve["successful_count"],
        "best_latency_ms_at_50": curve["best_latency_ms"],
        "oracle_latency_ms": result["oracle"]["best_latency_ms"],
        "oracle_statuses": result["oracle"]["statuses"],
    }


def geometric_mean(values):
    values = [value for value in values if value is not None and value > 0]
    return math.exp(sum(math.log(value) for value in values) / len(values)) if values else None


def aggregate(rows, method_name):
    supported = [row for row in rows if row["methods"][method_name]["supported"]]
    unsupported = [row["workload"] for row in rows if not row["methods"][method_name]["supported"]]
    successful = [row for row in supported if row["methods"][method_name]["successful"]]
    both = [row for row in supported if row["methods"][method_name]["within_50_percent_both_oracles"]]
    neither = [row["workload"] for row in supported if not row["methods"][method_name]["successful"]]
    one_only = [
        row["workload"]
        for row in supported
        if row["methods"][method_name]["successful"] and not row["methods"][method_name]["within_50_percent_both_oracles"]
    ]
    result = {
        "workloads": len(rows),
        "supported_workloads": len(supported),
        "unsupported_workloads": unsupported,
        "within_50_percent_either_oracle": len(successful),
        "within_50_percent_both_oracles": len(both),
        "outside_50_percent_either_oracle": neither,
        "outside_50_percent_both_oracles": [
            row["workload"] for row in supported if not row["methods"][method_name]["within_50_percent_both_oracles"]
        ],
        "exactly_one_oracle_within_50_percent": one_only,
        "unreachable_in_both_oracles": [row["workload"] for row in supported if not row["methods"][method_name]["oracle_reachable_either"]],
        "by_oracle": {},
        "by_family": {},
    }
    for oracle_name in ("E1", "E2"):
        oracles = [row["methods"][method_name]["oracles"][oracle_name] for row in supported]
        qualities = [oracle["oracle_at_50"] for oracle in oracles]
        result["by_oracle"][oracle_name] = {
            "exact_hits_within_50_percent": sum(
                oracle_pass(row["methods"][method_name]["oracles"][oracle_name], row["fifty_percent_budget"]) for row in supported
            ),
            "supported_workloads": len(supported),
            "unreachable_exact_oracle": sum(oracle["first_oracle_hit_k"] is None for oracle in oracles),
            "geometric_mean_oracle_at_50": geometric_mean(qualities),
            "worst_oracle_at_50": min((value for value in qualities if value is not None), default=None),
        }
    for family in dict.fromkeys(row["family"] for row in rows):
        family_rows = [row for row in rows if row["family"] == family]
        family_supported = [row for row in family_rows if row["methods"][method_name]["supported"]]
        result["by_family"][family] = {
            "workloads": len(family_rows),
            "supported_workloads": len(family_supported),
            "within_50_percent_either_oracle": sum(row["methods"][method_name]["successful"] for row in family_supported),
            "within_50_percent_both_oracles": sum(
                row["methods"][method_name]["within_50_percent_both_oracles"] for row in family_supported
            ),
        }
    return result


def bound_timing(rows):
    families = []
    for family in dict.fromkeys(row["family"] for row in rows):
        selected = [row for row in rows if row["family"] == family]
        families.append(
            {
                "family": family,
                "workloads": len(selected),
                "configurations": sum(row["bound_audit"]["scored_count"] for row in selected),
                "selected": sum(row["bound_audit"]["selected_count"] for row in selected),
                "occupancy_penalized": sum(
                    count
                    for row in selected
                    for penalty, count in row["bound_audit"]["occupancy_penalty_counts"].items()
                    if int(penalty) > 1
                ),
                "observed_would_reject": sum(row["bound_audit"]["observed_would_reject_count"] for row in selected),
                "audited_selection_seconds": sum(row["bound_audit"]["analysis_seconds"] for row in selected),
            }
        )
    keys = (
        "workloads",
        "configurations",
        "selected",
        "occupancy_penalized",
        "observed_would_reject",
        "audited_selection_seconds",
    )
    total = {key: sum(family[key] for family in families) for key in keys}
    return {
        "semantics": (
            "Sum of per-workload CPU worker elapsed times from the fresh bound-aware validation. "
            "Each interval includes candidate TIR construction, before/after TIR serialization mutation checks, "
            "analysis, ranking, and selection; it excludes compilation, GPU work, and oracle lookup."
        ),
        "families": families,
        "total": total,
    }


def compare_bound_aware_baseline(rows, baseline):
    """Compare pool-normalized rankings with the former classified metric."""
    baseline_rows = {row["workload"]: row for row in baseline["rows"]}
    if set(baseline_rows) != {row["workload"] for row in rows}:
        raise ValueError("baseline and current reports must contain the same workloads")
    ranking_changed = []
    oracle_tail_changes = []
    oracle_quality_changes = []
    for row in rows:
        workload = row["workload"]
        previous = baseline_rows[workload]
        old_report = read_json(previous["bound_audit"]["ranking"]["path"])
        new_report = read_json(row["bound_audit"]["ranking"]["path"])
        old_records = {record["index"]: record for record in old_report["configs"]}
        new_records = {record["index"]: record for record in new_report["configs"]}
        if old_records.keys() != new_records.keys():
            raise ValueError(f"baseline pool mismatch for {workload}")
        for index, old_record in old_records.items():
            new_record = new_records[index]
            if old_record["config"] != new_record["config"]:
                raise ValueError(f"baseline configuration mismatch for {workload}/{index}")
        # Scores use different pool-wide scales. Compare candidate order and
        # exact tie groups, not the raw numeric representation.
        rank_fields = ("index", "tier", "tie_first_rank", "tie_last_rank")
        old_ranking = [tuple(item.get(field) for field in rank_fields) for item in old_report["ranking"]]
        new_ranking = [tuple(item.get(field) for field in rank_fields) for item in new_report["ranking"]]
        if old_ranking != new_ranking:
            ranking_changed.append(workload)
        for oracle_name in ("E1", "E2"):
            old_oracle = previous["methods"]["bound_aware"]["oracles"][oracle_name]
            new_oracle = row["methods"]["bound_aware"]["oracles"][oracle_name]
            if old_oracle["first_oracle_hit_k"] != new_oracle["first_oracle_hit_k"]:
                oracle_tail_changes.append(f"{workload}/{oracle_name}")
            if old_oracle["oracle_at_50"] != new_oracle["oracle_at_50"]:
                oracle_quality_changes.append(f"{workload}/{oracle_name}")
    old_aggregate = baseline["aggregate"]["bound_aware"]
    new_aggregate = aggregate(rows, "bound_aware")
    return {
        "baseline_formula": "classified lexicographic (U * P, -D, E)",
        "current_formula": "lexicographic(max(U / max_pool(U), (U * P) / max_pool(U * P)), E, -D)",
        "ranking_changed_workloads": ranking_changed,
        "oracle_tail_changed_checks": oracle_tail_changes,
        "oracle_at_50_changed_checks": oracle_quality_changes,
        "either_oracle_success_before": old_aggregate["within_50_percent_either_oracle"],
        "either_oracle_success_after": new_aggregate["within_50_percent_either_oracle"],
        "both_oracle_success_before": old_aggregate["within_50_percent_both_oracles"],
        "both_oracle_success_after": new_aggregate["within_50_percent_both_oracles"],
        "performance_metric": "workloads whose E1 and E2 exact oracles are both in the strict top 50%",
        "performance_improved": (new_aggregate["within_50_percent_both_oracles"] > old_aggregate["within_50_percent_both_oracles"]),
    }


def compare_method_tail_ranks(rows, method, baseline):
    """Count per-oracle conservative tail-rank changes between two methods."""
    counts = {"better": 0, "equal": 0, "worse": 0}
    for row in rows:
        for oracle_name in ("E1", "E2"):
            current = row["methods"][method]["oracles"][oracle_name]["first_oracle_hit_k"]
            previous = row["methods"][baseline]["oracles"][oracle_name]["first_oracle_hit_k"]
            if current is None or previous is None:
                outcome = "equal" if current is previous else "worse" if current is None else "better"
            else:
                outcome = "better" if current < previous else "worse" if current > previous else "equal"
            counts[outcome] += 1
    return {
        "method": method,
        "baseline": baseline,
        "oracle_checks": sum(counts.values()),
        **counts,
    }


def pct(value):
    return "N/A" if value is None else f"{100 * value:.2f}%"


def tail(oracle, pool_size):
    rank = oracle["first_oracle_hit_k"]
    return "unreachable" if rank is None else f"{rank} ({100 * rank / pool_size:.2f}%)"


def render(summary):
    aggregates = summary["aggregate"]
    timing = summary["timing"]
    old_timing = timing["prior_report"]
    bound = timing["bound_aware_audit"]
    lines = [
        "# H200 cost-model comparison with normalized bound-aware across 25 workloads",
        "",
        (
            "This comparison extends the former 25-workload report with fresh current-IR TileTune bound-aware rankings. "
            "All four methods are joined by configuration identity to the same saved exhaustive E1/E2 measurements; "
            "no held-out candidate is recompiled or rebenchmarked for this report."
        ),
        "",
        "## Main result",
        "",
        (
            "- A workload is marked successful when **at least one of its E1 or E2 exact optima** has conservative "
            "tail rank at or below `floor(pool_size / 2)`."
        ),
    ]
    for method_name in METHODS:
        item = aggregates[method_name]
        lines.append(
            f"- {LABELS[method_name]} succeeds on **{item['within_50_percent_either_oracle']}/{item['workloads']}** "
            f"workloads under that rule ({item['within_50_percent_both_oracles']}/{item['workloads']} pass both oracles)."
        )
    lines += [
        (
            "- Normalized bound-aware retains both E1 and E2 exact optima within the strict 50% budget on all 25 workloads."
        ),
        "- Bound-aware scores all **14,425/14,425** configurations and uses no pre-lowering or post-compile exclusion.",
        (
            f"- Versus the classified baseline, pool normalization changes "
            f"**{len(summary['baseline_comparison']['ranking_changed_workloads'])}** workload rankings and "
            f"**{len(summary['baseline_comparison']['oracle_tail_changed_checks'])}** oracle tails; "
            f"both-oracle retention improves from **{summary['baseline_comparison']['both_oracle_success_before']}/25** "
            f"to **{summary['baseline_comparison']['both_oracle_success_after']}/25**."
        ),
        (
            f"- Versus TileTune memory, normalized bound-aware improves "
            f"**{summary['memory_comparison']['better']}/50** oracle tail ranks, ties "
            f"**{summary['memory_comparison']['equal']}/50**, and worsens **{summary['memory_comparison']['worse']}/50**."
        ),
        "",
        (
            "| Method | Support | E1 exact within 50% | E1 geo Oracle@50 | E1 worst | "
            "E2 exact within 50% | E2 geo Oracle@50 | E2 worst | Either-oracle success | Both-oracle diagnostic |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method_name in METHODS:
        item = aggregates[method_name]
        e1, e2 = item["by_oracle"]["E1"], item["by_oracle"]["E2"]
        lines.append(
            f"| {LABELS[method_name]} | {item['supported_workloads']}/{item['workloads']} | "
            f"{e1['exact_hits_within_50_percent']}/{e1['supported_workloads']} | "
            f"{pct(e1['geometric_mean_oracle_at_50'])} | {pct(e1['worst_oracle_at_50'])} | "
            f"{e2['exact_hits_within_50_percent']}/{e2['supported_workloads']} | "
            f"{pct(e2['geometric_mean_oracle_at_50'])} | {pct(e2['worst_oracle_at_50'])} | "
            f"{item['within_50_percent_either_oracle']}/{item['workloads']} | "
            f"{item['within_50_percent_both_oracles']}/{item['workloads']} |"
        )

    lines += [
        "",
        "## Bound-aware score and audit contract",
        "",
        "Each candidate first receives two semantic work quantities:",
        "",
        "`U` and `O = U * P`",
        "",
        (
            "`U` is launch-underfill-adjusted logical byte-waves, `D` is pipeline depth, and `E` is logical "
            "memory-access waves. Every candidate uses `P = ceil(8 / active_warps_per_SM)`, clamped to at least 1."
        ),
        "",
        (
            "After all candidates are analyzed, the final ranking is "
            "`(max(U / max_pool(U), O / max_pool(O)), E, -D)`. Exact common-denominator numerators preserve ties."
        ),
        "",
        (
            "No compute-versus-memory classifier or roofline ridge enters the score. The occupancy-adjusted component "
            "is a coarse schedule proxy; it changes ordering only and does not reject configurations."
        ),
        "",
        (
            "The validation omits the pre-lowering resource decision from every ranking record and sets "
            "`max_spill_bytes=None`, `max_local_bytes=None`, and `post_compile_policy=None`. Observed resource pressure "
            "is audit data only and never changes eligibility."
        ),
        "",
        "## Time cost",
        "",
        (
            "The XGBoost, TileTune memory, and Carver figures below are carried over unchanged from the former report. "
            "XGBoost label collection includes correctness-checked compilation and benchmarking; oracle lookup and prior "
            "exhaustive collection are excluded."
        ),
        "",
        (
            "| Family | Train success | Train labels s | Validation success | Validation labels s | "
            "Fit / early-stop s | Load + held-out rank s | XGB charged total s |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family in old_timing["families"]:
        lines.append(
            f"| {family['family']} | {family['training_successes']}/{family['training_attempts']} | "
            f"{family['training_label_collection_seconds']:.3f} | "
            f"{family['validation_successes']}/{family['validation_attempts']} | "
            f"{family['validation_label_collection_seconds']:.3f} | {family['fit_early_stop_seconds']:.3f} | "
            f"{family['model_load_heldout_ranking_seconds']:.3f} | {family['xgboost_charged_seconds']:.3f} |"
        )
    total = old_timing["total"]
    lines.append(
        f"| **Total** | **{total['training_successes']}/{total['training_attempts']}** | "
        f"**{total['training_label_collection_seconds']:.3f}** | "
        f"**{total['validation_successes']}/{total['validation_attempts']}** | "
        f"**{total['validation_label_collection_seconds']:.3f}** | "
        f"**{total['fit_early_stop_seconds']:.3f}** | "
        f"**{total['model_load_heldout_ranking_seconds']:.3f}** | **{total['xgboost_charged_seconds']:.3f}** |"
    )
    by_family_bound = {row["family"]: row for row in bound["families"]}
    lines += [
        "",
        (
            "TileTune memory timing has the former report's three accounting policies. Bound-aware is shown separately "
            "as audited CPU-worker time because its validation also serializes every TIR before and after analysis to "
            "check non-mutation, so it is not a production model-only timing."
        ),
        "",
        (
            "| Family | XGB charged s | TileTune memory measured s | Bound-aware audited s | "
            "Memory reuse-adjusted s | Memory model-only s | Carver rank s |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family in old_timing["families"]:
        audit = by_family_bound[family["family"]]
        lines.append(
            f"| {family['family']} | {family['xgboost_charged_seconds']:.3f} | "
            f"{family['tiletune_measured_selection_seconds']:.3f} | {audit['audited_selection_seconds']:.3f} | "
            f"{family['tiletune_reuse_adjusted_incremental_seconds']:.3f} | "
            f"{family['tiletune_model_only_preelaborated_seconds']:.3f} | {family['carver_ranking_seconds']:.3f} |"
        )
    bound_total = bound["total"]
    lines.append(
        f"| **Total** | **{total['xgboost_charged_seconds']:.3f}** | "
        f"**{total['tiletune_measured_selection_seconds']:.3f}** | "
        f"**{bound_total['audited_selection_seconds']:.3f}** | "
        f"**{total['tiletune_reuse_adjusted_incremental_seconds']:.3f}** | "
        f"**{total['tiletune_model_only_preelaborated_seconds']:.3f}** | "
        f"**{total['carver_ranking_seconds']:.3f}** |"
    )

    lines += [
        "",
        "### Bound-aware audit composition",
        "",
        "| Family | Configs / selected | Occupancy-penalized | Would-reject observed | Audited s |",
        "|---|---:|---:|---:|---:|",
    ]
    for family in bound["families"]:
        lines.append(
            f"| {family['family']} | {family['configurations']:,} / {family['selected']:,} | "
            f"{family['occupancy_penalized']:,} | {family['observed_would_reject']:,} | "
            f"{family['audited_selection_seconds']:.3f} |"
        )
    lines.append(
        f"| **Total** | **{bound_total['configurations']:,} / {bound_total['selected']:,}** | "
        f"**{bound_total['occupancy_penalized']:,}** | **{bound_total['observed_would_reject']:,}** | "
        f"**{bound_total['audited_selection_seconds']:.3f}** |"
    )
    lines += [
        "",
        (
            "The selected total can be smaller than the nominal 7,210 half-pool budget because strict selection "
            "excludes an entire equal-score group when that group would cross the boundary."
        ),
    ]

    lines += [
        "",
        "## E1/E2 conservative oracle tail rank per workload",
        "",
        (
            "Each method cell is `E1 / E2`. A rank is successful when it is no larger than the strict 50% K. "
            "The final cell for each method applies the requested either-oracle rule."
        ),
        "",
        (
            "| Workload | Family | Pool / 50% K | Memory E1 / E2 | Memory success | "
            "Bound-aware E1 / E2 | Bound-aware success | XGBoost E1 / E2 | XGBoost success | "
            "Carver E1 / E2 | Carver success |"
        ),
        "|---|---|---:|---:|:---:|---:|:---:|---:|:---:|---:|:---:|",
    ]
    for row in summary["rows"]:
        values = []
        for method_name in METHODS:
            method = row["methods"][method_name]
            values += [
                f"{tail(method['oracles']['E1'], row['pool_size'])} / {tail(method['oracles']['E2'], row['pool_size'])}",
                "yes" if method["successful"] else "NO",
            ]
        lines.append(
            f"| {row['workload']} | {row['family']} | {row['pool_size']} / {row['fifty_percent_budget']} | " + " | ".join(values) + " |"
        )

    lines += [
        "",
        "## Carver model correction",
        "",
        (
            "- FP8 GEMM uses the existing legacy Carver matmul policy with the restored kernel's exact graph: "
            "unscaled E4M3 operands, FP32 accumulation, and E4M3 output."
        ),
        (
            "- KDA uses the experiment-side adapter derived from the active token-parallel intra kernel. "
            "Neither adapter is fitted to E1/E2 outcomes."
        ),
        "",
        "## Validation and interpretation",
        "",
        "- Rankings are frozen before the corresponding E1/E2 outcome table is opened for comparison.",
        "- Equal primary-score groups use conservative tail rank; failures consume budget and are not replaced.",
        "- The requested success result uses E1-or-E2. The stricter both-oracle result remains in the main table as a diagnostic.",
        "- Bound-aware ranks the complete active pool and applies no post-compile filter or resource-based eligibility filter.",
        "- The report evaluates 200 method/oracle joins: 25 workloads x 4 methods x 2 oracles.",
        "- Oracle@50 uses saved exhaustive latencies and is not a fresh winner remeasurement.",
        "- This remains a retrospective fixed-pool validation and does not establish out-of-sample generalization.",
        "",
        (
            "Machine-readable details are in `summary.json`, `workloads.csv`, and the 50 new bound-aware files "
            "under `comparisons/`. Prior ranking and timing artifacts are referenced by path and SHA256 rather than copied."
        ),
        "",
    ]
    return "\n".join(lines)


def write_csv(path, rows):
    fields = ["workload", "family", "op", "pool_size", "fifty_percent_budget"]
    for method_name in METHODS:
        fields += [
            f"{method_name}_status",
            f"{method_name}_e1_first_oracle_hit_k",
            f"{method_name}_e2_first_oracle_hit_k",
            f"{method_name}_within_50_percent_either_oracle",
            f"{method_name}_within_50_percent_both_oracles",
            f"{method_name}_e1_oracle_percent_at_50",
            f"{method_name}_e2_oracle_percent_at_50",
        ]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in fields[:5]}
            for method_name in METHODS:
                method = row["methods"][method_name]
                flat.update(
                    {
                        f"{method_name}_status": method["status"],
                        f"{method_name}_e1_first_oracle_hit_k": method["oracles"]["E1"]["first_oracle_hit_k"],
                        f"{method_name}_e2_first_oracle_hit_k": method["oracles"]["E2"]["first_oracle_hit_k"],
                        f"{method_name}_within_50_percent_either_oracle": method["within_50_percent_either_oracle"],
                        f"{method_name}_within_50_percent_both_oracles": method["within_50_percent_both_oracles"],
                        f"{method_name}_e1_oracle_percent_at_50": method["oracles"]["E1"]["oracle_percent_at_50"],
                        f"{method_name}_e2_oracle_percent_at_50": method["oracles"]["E2"]["oracle_percent_at_50"],
                    }
                )
            writer.writerow(flat)


def build(prior_dir, bound_dir, baseline_dir, output):
    prior_dir, bound_dir, baseline_dir, output = map(lambda path: Path(path).resolve(), (prior_dir, bound_dir, baseline_dir, output))
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    prior_path, bound_path = prior_dir / "summary.json", bound_dir / "summary.json"
    prior, bound_summary = read_json(prior_path), read_json(bound_path)
    baseline_path = baseline_dir / "summary.json"
    baseline = read_json(baseline_path)
    old_by_name = {row["workload"]: row for row in prior["rows"]}
    bound_by_name = {row["workload"]: row for row in bound_summary["rows"]}
    if len(old_by_name) != 25 or old_by_name.keys() != bound_by_name.keys():
        raise ValueError("prior and bound-aware reports must contain the same 25 unique workloads")
    if bound_summary.get("post_compile_filter") is not False or bound_summary.get("pre_lowering_decisions_used_for_ranking") is not False:
        raise ValueError("bound-aware input must not use pre-lowering or post-compile filtering")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        rows = []
        for old_row in prior["rows"]:
            workload = old_row["workload"]
            bound_row = bound_by_name[workload]
            if old_row["pool_size"] != bound_row["pool_size"] or old_row["fifty_percent_budget"] != bound_row["alpha_budget"]:
                raise ValueError(f"pool or budget mismatch for {workload}")
            methods = {
                "tiletune_memory": derive_method(old_row["methods"]["tiletune"], old_row["fifty_percent_budget"], old_row["pool_size"]),
                "xgboost": derive_method(old_row["methods"]["xgboost"], old_row["fifty_percent_budget"], old_row["pool_size"]),
                "carver": derive_method(old_row["methods"]["carver"], old_row["fifty_percent_budget"], old_row["pool_size"]),
            }
            bound_oracles = {}
            for oracle_name in ("E1", "E2"):
                result = compare(
                    bound_row["oracles"][oracle_name]["source"]["path"],
                    {"bound_aware": bound_row["ranking"]["path"]},
                    [bound_row["alpha_budget"]],
                )
                relative = Path("comparisons") / oracle_name.lower() / f"{workload}.json"
                staged_path, final_path = staging / relative, output / relative
                staged_path.parent.mkdir(parents=True, exist_ok=True)
                write_json(staged_path, result)
                bound_oracles[oracle_name] = comparison_oracle(result, staged_provenance(staged_path, final_path))
            methods["bound_aware"] = derive_method(
                {
                    "oracles": bound_oracles,
                    "status": "evaluated",
                    "supported": True,
                },
                old_row["fifty_percent_budget"],
                old_row["pool_size"],
            )
            rows.append(
                {
                    "workload": workload,
                    "family": old_row["family"],
                    "op": old_row["op"],
                    "pool_size": old_row["pool_size"],
                    "fifty_percent_budget": old_row["fifty_percent_budget"],
                    "bound_audit": {
                        key: copy.deepcopy(bound_row[key])
                        for key in (
                            "ranking",
                            "scored_count",
                            "eligible_count",
                            "selected_count",
                            "occupancy_penalty_counts",
                            "observed_would_reject_count",
                            "analysis_seconds",
                        )
                    },
                    "methods": {method_name: methods[method_name] for method_name in METHODS},
                }
            )

        aggregates = {method_name: aggregate(rows, method_name) for method_name in METHODS}
        bound_times = bound_timing(rows)
        baseline_comparison = compare_bound_aware_baseline(rows, baseline)
        memory_comparison = compare_method_tail_ranks(rows, "bound_aware", "tiletune_memory")
        summary = {
            "version": 2,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scope": (
                "Twenty-five H200 held-out workloads; prior TileTune memory, XGBoost, and Carver results "
                "plus fresh current bound-aware rankings, all joined to saved exhaustive E1/E2 artifacts."
            ),
            "semantics": {
                "comparison": (
                    "Rankings are joined by configuration identity to E1/E2. No held-out configuration is recompiled or rebenchmarked."
                ),
                "success": (
                    "A workload succeeds when at least one of E1 or E2 has an exact optimum with conservative "
                    "tail rank <= floor(pool_size / 2)."
                ),
                "both_oracle_diagnostic": "The former both-E1-and-E2 result is retained as a stricter diagnostic only.",
                "oracle_at_50": (
                    "Oracle best latency divided by the fastest successful oracle latency in complete primary-score "
                    "groups within floor(pool_size / 2)."
                ),
                "bound_aware": (
                    "Pool-normalized max of memory and occupancy work with `(E, -D)` tie-breaks; all scored configurations "
                    "eligible; no pre-lowering or post-compile filter."
                ),
            },
            "aggregate": aggregates,
            "bound_aware_model": {
                "score_formula": bound_summary["score_formula"],
                "classifier": "not_used",
            },
            "baseline_comparison": baseline_comparison,
            "memory_comparison": memory_comparison,
            "timing": {
                "prior_report": prior["timing"],
                "bound_aware_audit": bound_times,
            },
            "rows": rows,
            "sources": {
                "prior_report": provenance(prior_path),
                "bound_aware_validation": provenance(bound_path),
                "bound_aware_classified_baseline": provenance(baseline_path),
                "prior_sources": prior["sources"],
                "bound_aware_code": bound_summary.get("code"),
            },
            "checks": {
                "workload_count_25": len(rows) == 25,
                "method_oracle_entries_200": len(rows) * len(METHODS) * 2 == 200,
                "all_methods_supported": all(row["methods"][name]["supported"] for row in rows for name in METHODS),
                "all_pool_sizes_match": all(row["pool_size"] == row["bound_audit"]["scored_count"] for row in rows),
                "bound_aware_all_configs_eligible": all(row["bound_audit"]["eligible_count"] == row["pool_size"] for row in rows),
                "bound_aware_no_post_compile_filter": bound_summary["post_compile_filter"] is False,
                "bound_aware_no_pre_lowering_filter": bound_summary["pre_lowering_decisions_used_for_ranking"] is False,
                "tiletune_memory_success_25": aggregates["tiletune_memory"]["within_50_percent_either_oracle"] == 25,
                "bound_aware_success_25": aggregates["bound_aware"]["within_50_percent_either_oracle"] == 25,
                "bound_aware_both_oracles_25": aggregates["bound_aware"]["within_50_percent_both_oracles"] == 25,
                "xgboost_success_20": aggregates["xgboost"]["within_50_percent_either_oracle"] == 20,
                "carver_success_11": aggregates["carver"]["within_50_percent_either_oracle"] == 11,
            },
        }
        if not all(summary["checks"].values()):
            raise AssertionError({key: value for key, value in summary["checks"].items() if not value})
        write_json(staging / "summary.json", summary)
        write_csv(staging / "workloads.csv", rows)
        (staging / "REPORT.md").write_text(render(summary))
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging)
        raise
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--bound-aware", type=Path, default=DEFAULT_BOUND_AWARE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        summary = build(args.prior, args.bound_aware, args.baseline, args.output)
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
        parser.error(str(error))
    print(f"wrote {args.output.resolve()}")
    for method_name in METHODS:
        item = summary["aggregate"][method_name]
        print(
            f"{LABELS[method_name]}: either {item['within_50_percent_either_oracle']}/{item['workloads']}; "
            f"both {item['within_50_percent_both_oracles']}/{item['workloads']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
