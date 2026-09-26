"""Validate normalized memory/occupancy ranking against H200 E1/E2 oracles.

Every current configuration is freshly analyzed before any measured latency is
opened. Resource decisions are recorded for audit only: ranking records carry
no pre-lowering decision, and this CPU-only validation performs no compilation
or post-compile filtering. Equal scores use their conservative group tail rank.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time

from experiments.utils.results import config_key, provenance


ROOT = Path(__file__).resolve().parents[1]
TARGET = {"kind": "cuda", "arch": "sm_90a"}
H200_LIMITS = {
    "sm_count": 132,
    "shared_memory_per_sm": 233472,
    "shared_memory_per_block": 232448,
    "registers_per_sm": 65536,
    "max_threads_per_sm": 2048,
    "max_threads_per_block": 1024,
    "max_blocks_per_sm": 32,
    "warp_size": 32,
}
ORACLE_ROOTS = {
    "E1": ROOT / "experiments/results/h200-e1-event-20260922T210645Z",
    "E2": ROOT / "experiments/results/h200-e2-event-20260922T174514Z",
}
GEMM_ORACLE_ROOTS = {
    "E1": ROOT / "experiments/results/h200-gemm-576-e1-20260924",
    "E2": ROOT / "experiments/results/h200-gemm-576-e2-20260924",
}
TERMINAL = {
    "benchmarked",
    "compilation_failed",
    "elaboration_failed",
    "analysis_failed",
    "pre_lowering_rejected",
    "post_compile_rejected",
    "benchmark_error",
    "benchmark_timeout",
    "worker_failed",
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def completed_outcomes(root, workload):
    """Read the one completed exhaustive attempt named by its root manifest."""
    root = Path(root).resolve()
    rows = json.loads((root / "comparison.json").read_text())
    matches = [row for row in rows if row.get("workload") == workload and row.get("status") == "completed"]
    if len(matches) != 1:
        raise ValueError(f"{root}: expected one completed {workload} result")
    path = root / matches[0]["attempt"] / "outcomes.json"
    return path, json.loads(path.read_text())


def analyze_workload(workload, configs, output, alpha):
    """Freshly rank one complete pool without opening an oracle."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # The global editable finder may name another worktree. Prefer this source
    # checkout before importing TileLang in the worker.
    sys.meta_path = [finder for finder in sys.meta_path if "ScikitBuildRedirectingFinder" not in type(finder).__name__]
    from experiments.common.kernels import make_case
    from experiments.common.spec import Workload
    from tilelang.tiletune import analyze_prim_func, TileTuneConfig
    from tiletune_core.ranking import alpha_budget, rank_records, select_top_k

    workload = Workload(**workload)
    case = make_case(workload)
    settings = TileTuneConfig(
        enabled=True,
        mode="report_only",
        ranking_metric="bound_aware",
        alpha=alpha,
        strict_top_k=True,
        memory_diagnostics=False,
        input_values=case.input_values or None,
        device_limits=H200_LIMITS,
        max_spill_bytes=None,
        max_local_bytes=None,
        post_compile_policy=None,
    )
    records = []
    penalty_counts = Counter()
    would_reject = 0
    started = time.perf_counter()
    for index, config in enumerate(configs):
        func = case.build(**config)
        before = func.script()
        result = analyze_prim_func(
            func,
            settings,
            target=TARGET,
            device_limits=H200_LIMITS,
            pass_configs=case.pass_configs,
        )
        if func.script() != before:
            raise AssertionError(f"analysis changed PrimFunc {workload.name}/{index}")
        bound = result["modules"]["bound"]
        penalty_counts[str(bound["occupancy_penalty"])] += 1
        would_reject += bool(result["pressure"]["decision"].get("would_reject"))
        # Resource decisions are deliberately omitted from the ranking record.
        # This makes every scored configuration eligible, independently of any
        # pre- or post-compile resource observation.
        records.append(
            {
                "index": index,
                "config": config,
                "status": "analyzed",
                "pre_lowering": None,
                "tile_cost": result["tile_cost"],
                "bound": bound,
            }
        )
    ranking = rank_records(records)
    budget = alpha_budget(len(records), alpha)
    selected = select_top_k(ranking, budget, strict_budget=True)
    elapsed = time.perf_counter() - started
    path = Path(output) / workload.name / "bound-aware.json"
    path.parent.mkdir(parents=True, exist_ok=False)
    write_json(
        path,
        {
            "version": 3,
            "workload": workload.to_dict(),
            "target": TARGET,
            "device_limits": H200_LIMITS,
            "settings": settings.to_cache_key_dict(),
            "measurement_scope": "fresh CPU IR analysis; no compilation, GPU execution, or oracle input",
            "selection_policy": "all scored configurations eligible; no pre-lowering or post-compile resource filter",
            "configs": records,
            "ranking": ranking,
            "selection": {
                "alpha": alpha,
                "pool_size": len(records),
                "requested_k": budget,
                "strict_budget": True,
                "selected_indices": selected,
                "selected_count": len(selected),
                "tie_policy": "exclude_boundary_score_group",
            },
            "occupancy_penalty_counts": dict(penalty_counts),
            "score_formula": (
                "lexicographic(max(adjusted_logical_byte_waves / max_pool_adjusted_logical_byte_waves, "
                "occupancy_adjusted_logical_byte_waves / max_pool_occupancy_adjusted_logical_byte_waves), "
                "logical_memory_access_waves, -pipeline_depth)"
            ),
            "observed_would_reject_count": would_reject,
            "analysis_seconds": elapsed,
        },
    )
    return {
        "workload": workload.name,
        "family": workload.op,
        "ranking": provenance(path),
        "pool_size": len(records),
        "scored_count": sum(row["tier"] == "eligible" for row in ranking),
        "eligible_count": sum(row["tier"] == "eligible" for row in ranking),
        "selected_count": len(selected),
        "alpha_budget": budget,
        "occupancy_penalty_counts": dict(penalty_counts),
        "observed_would_reject_count": would_reject,
        "analysis_seconds": elapsed,
    }


def evaluate_oracle(report, root):
    """Join one frozen ranking to an exhaustive oracle after ranking is fixed."""
    ranking_by_index = {row["index"]: row for row in report["ranking"]}
    records_by_config = {config_key(row["config"]): row for row in report["configs"]}
    if len(records_by_config) != len(report["configs"]):
        raise ValueError("ranking pool contains duplicate configurations")
    outcomes_path, outcomes = completed_outcomes(root, report["workload"]["name"])
    filtered = [row for row in outcomes if config_key(row["config"]) in records_by_config]
    if len(filtered) != len(records_by_config) or len({config_key(row["config"]) for row in filtered}) != len(filtered):
        raise ValueError(f"{outcomes_path}: oracle does not contain every active configuration exactly once")
    if any(row.get("status") not in TERMINAL for row in filtered):
        raise ValueError(f"{outcomes_path}: oracle subset contains a nonterminal outcome")
    valid = [row for row in filtered if row["status"] == "benchmarked"]
    if not valid or any(
        type(row.get("latency_ms")) not in (int, float) or not math.isfinite(row["latency_ms"]) or row["latency_ms"] <= 0 for row in valid
    ):
        raise ValueError(f"{outcomes_path}: no valid finite oracle measurements")
    best = min(row["latency_ms"] for row in valid)
    winners = [row for row in valid if row["latency_ms"] == best]
    selected = set(report["selection"]["selected_indices"])
    candidates = []
    for outcome in winners:
        record = records_by_config[config_key(outcome["config"])]
        rank = ranking_by_index[record["index"]]
        candidates.append(
            {
                "index": record["index"],
                "source_index": outcome["index"],
                "config": outcome["config"],
                "latency_ms": outcome["latency_ms"],
                "score": rank["score"],
                "combined_primary_numerator": rank["combined_primary_numerator"],
                "combined_primary_denominator": rank["combined_primary_denominator"],
                "normalized_score": rank["normalized_score"],
                "normalized_memory_work": rank["normalized_memory_work"],
                "normalized_occupancy_work": rank["normalized_occupancy_work"],
                "tie_first_rank": rank["tie_first_rank"],
                "tie_last_rank": rank["tie_last_rank"],
                "selected_at_alpha": record["index"] in selected,
                "occupancy_penalty": record["bound"]["occupancy_penalty"],
            }
        )
    tail = min(candidate["tie_last_rank"] for candidate in candidates)
    budget = report["selection"]["requested_k"]
    return {
        "source": provenance(outcomes_path),
        "source_pool_size": len(outcomes),
        "pool_size": len(filtered),
        "statuses": dict(Counter(row["status"] for row in filtered)),
        "oracle_latency_ms": best,
        "oracle_count": len(candidates),
        "oracle_candidates": candidates,
        "best_tail_rank": tail,
        "tail_fraction": tail / len(filtered),
        "alpha_budget": budget,
        "within_alpha": tail <= budget,
    }


def render(summary):
    lines = [
        "# H200 bound-aware oracle-retention validation",
        "",
        "Fresh current-IR analysis ranks every configuration in each active pool before E1/E2 latency labels are opened. All scored configurations are eligible: pre-lowering resource decisions are ignored for ranking, and no compilation or post-compile filter is run.",
        "",
        "The primary score is the maximum of pool-normalized memory and occupancy-adjusted byte-waves. Access-waves and descending pipeline depth break exact primary ties; no roofline classification is used.",
        "",
        f"Result: **{summary['passed_checks']}/{summary['total_checks']} workload/oracle checks preserve an oracle within the strict top {100 * summary['alpha']:.0f}%**.",
        "",
        "| Workload | Pool | Scored | Occupancy-penalized | Oracle | Tail rank | Pool share | <= 50% |",
        "|---|---:|---:|---:|---|---:|---:|:---:|",
    ]
    for row in summary["rows"]:
        penalized = sum(count for penalty, count in row["occupancy_penalty_counts"].items() if int(penalty) > 1)
        for name, oracle in row["oracles"].items():
            lines.append(
                f"| {row['workload']} | {row['pool_size']} | {row['scored_count']} | {penalized} | {name} | "
                f"{oracle['best_tail_rank']}/{oracle['alpha_budget']} | {100 * oracle['tail_fraction']:.2f}% | "
                f"{'yes' if oracle['within_alpha'] else 'NO'} |"
            )
    lines += [
        "",
        f"Worst case: **{summary['worst_workload']} {summary['worst_oracle']}**, tail rank "
        f"**{summary['worst_tail_rank']}/{summary['worst_pool_size']} ({100 * summary['worst_tail_fraction']:.2f}%)**.",
        "",
        "This is a retrospective fixed-pool validation against existing exhaustive measurements. It performs no new GPU measurement and does not establish out-of-sample generalization.",
        "",
    ]
    return "\n".join(lines)


def validate(output, alpha, workers):
    from experiments.common.spec import Device, configurations, default_workloads

    output = Path(output).resolve()
    if output.exists():
        raise ValueError("output already exists; use a new result directory")
    output.mkdir(parents=True)
    device = Device(name="H200", target=TARGET, device_limits=H200_LIMITS)
    workloads = default_workloads()
    inputs = [(workload.to_dict(), configurations(workload, device)) for workload in workloads]
    rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as executor:
        futures = {executor.submit(analyze_workload, workload, configs, output, alpha): workload["name"] for workload, configs in inputs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['workload']}: analyzed {row['scored_count']}/{row['pool_size']}; "
                f"eligible {row['eligible_count']}; selected {row['selected_count']}/{row['alpha_budget']}",
                flush=True,
            )
            write_json(output / "progress.json", rows)

    order = {workload.name: index for index, workload in enumerate(workloads)}
    for row in rows:
        report = json.loads(Path(row["ranking"]["path"]).read_text())
        roots = GEMM_ORACLE_ROOTS if row["family"] == "gemm" else ORACLE_ROOTS
        row["oracles"] = {name: evaluate_oracle(report, root) for name, root in roots.items()}
        print(
            row["workload"]
            + ": "
            + ", ".join(f"{name} tail {oracle['best_tail_rank']}/{oracle['alpha_budget']}" for name, oracle in row["oracles"].items()),
            flush=True,
        )
    rows.sort(key=lambda row: order[row["workload"]])
    checks = [(row, name, oracle) for row in rows for name, oracle in row["oracles"].items()]
    worst_row, worst_name, worst = max(checks, key=lambda item: item[2]["tail_fraction"])
    summary = {
        "version": 3,
        "semantics": "fresh pool-normalized adjusted-byte/occupancy ranking with (E, -D) tie-breaks; no bound classifier; all scored configs eligible; no pre-lowering or post-compile filter; saved E1/E2 labels joined afterward",
        "target": TARGET,
        "device_limits": H200_LIMITS,
        "score_formula": (
            "lexicographic(max(adjusted_logical_byte_waves / max_pool_adjusted_logical_byte_waves, "
            "occupancy_adjusted_logical_byte_waves / max_pool_occupancy_adjusted_logical_byte_waves), "
            "logical_memory_access_waves, -pipeline_depth)"
        ),
        "alpha": alpha,
        "workload_count": len(rows),
        "configuration_count": sum(row["pool_size"] for row in rows),
        "scored_count": sum(row["scored_count"] for row in rows),
        "eligible_count": sum(row["eligible_count"] for row in rows),
        "selected_count": sum(row["selected_count"] for row in rows),
        "post_compile_filter": False,
        "pre_lowering_decisions_used_for_ranking": False,
        "rows": rows,
        "total_checks": len(checks),
        "passed_checks": sum(oracle["within_alpha"] for _, _, oracle in checks),
        "all_within_alpha": all(oracle["within_alpha"] for _, _, oracle in checks),
        "worst_workload": worst_row["workload"],
        "worst_oracle": worst_name,
        "worst_tail_rank": worst["best_tail_rank"],
        "worst_pool_size": worst["pool_size"],
        "worst_tail_fraction": worst["tail_fraction"],
        "total_analysis_seconds": sum(row["analysis_seconds"] for row in rows),
        "gpu_work": False,
        "code": [
            provenance(ROOT / path)
            for path in (
                "tilelang/tiletune/engine.py",
                "tilelang/tiletune/memory.py",
                "tiletune_core/memory.py",
                "tiletune_core/ranking.py",
                "experiments/validate_bound_aware.py",
            )
        ],
    }
    write_json(output / "summary.json", summary)
    (output / "REPORT.md").write_text(render(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    try:
        result = validate(args.output, args.alpha, args.workers)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(
        f"Completed {args.output / 'REPORT.md'}: {result['passed_checks']}/{result['total_checks']} checks within alpha",
        flush=True,
    )
    return 0 if result["all_within_alpha"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
