"""Re-score frozen TileTune facts without rebuilding kernels or using a GPU.

This replay supports the sealed multi-family study layout produced by
``experiments.common.run``. Scores are frozen before the exhaustive oracle is
opened, so measurements cannot influence ordering.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from math import prod
from pathlib import Path
import time

from tiletune_core.memory import score_memory
from tiletune_core.ranking import rank_records


# Backward-compatible interpretation of opaque operations in the archived B200
# reports. Current live analysis owns the same list in tilelang.tiletune.memory.
MEMORY_NEUTRAL_UNKNOWN_OPERATIONS = frozenset(
    {
        "tirx.assume",
        "tirx.ptx_arrive_barrier",
        "tirx.ptx_arrive_cluster_barrier",
        "tl.ptx_arrive_barrier",
        "tl.ptx_arrive_cluster_barrier",
        "tl.tcgen05_mma_arrive",
        "tl.ptx_tcgen05_cp_warpx4",
        "tl.ptx_tcgen05_sf_warp_transpose",
        "tl.fence_proxy_async",
    }
)


def read(path):
    return json.loads(Path(path).read_text())


def literal_int(value):
    """Archived expressions are data; accept integer literals only."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def memory_inputs(record, sm_count):
    """Recover the live memory adapter's inputs from an archived report row."""
    storage = {}
    for item in record["pressure"]["logical_storage"]:
        if item["scope"] != "global":
            continue
        if item["buffer"] in storage:
            raise ValueError("archived global buffer names must be unique")
        storage[item["buffer"]] = item
    archived_memory = record["modules"]["memory_traffic"]
    accesses = []
    unknown = []
    for op in record["tile_propagation"]["operations"]:
        if op["unknown"] and op["kind"] not in MEMORY_NEUTRAL_UNKNOWN_OPERATIONS:
            unknown.append(f"operation {op['index']}: unresolved memory effects")
        external = [
            (direction, region)
            for direction in ("reads", "writes")
            for region in op[direction]
            if region["scope"] == "global"
        ]
        if not external:
            continue
        loop_extents = [literal_int(loop["extent"]) for loop in op["loops"] if loop["kind"] != "4"]
        if all(value is not None and value >= 0 for value in loop_extents):
            visits = prod(loop_extents)
        else:
            bounds = {
                item["visits_per_block"]
                for item in archived_memory["input_tiles"]
                if item.get("operation") == op["index"]
            }
            if not bounds:
                buffers = {region["buffer"] for _, region in external}
                loop_variables = {loop["var"] for loop in op["loops"] if loop["kind"] != "4"}
                bounds = {
                    item["visits_per_block"]
                    for item in archived_memory["input_tiles"]
                    if item["buffer"] in buffers and set(item.get("loop_variables", ())) == loop_variables
                }
            if len(bounds) != 1 or None in bounds:
                raise ValueError(f"operation {op['index']}: no unambiguous archived loop-visit bound")
            visits = bounds.pop()
        for direction, region in external:
            item = storage[region["buffer"]]
            bits, elements = item["logical_bits"], item["logical_elements"]
            if bits is None or not elements or bits % elements:
                raise ValueError("archived storage does not resolve dtype width")
            extents = [literal_int(axis["extent"]) for axis in region["ranges"]]
            volume = prod(extents) if all(value is not None and value >= 0 for value in extents) else None
            accesses.append(
                {
                    "operation": op["index"],
                    "direction": direction,
                    "buffer": region["buffer"],
                    "bytes": (volume * (bits // elements) + 7) // 8 if volume is not None else None,
                    "visits": visits,
                }
            )
    waves = record["modules"]["waves"]
    shared = record["modules"]["memory_traffic"].get("shared_allocations", ())
    pipeline_depth = record["modules"]["pipeline_overlap"].get("effective_buffer_depth") or max(
        [1, *(item.get("pipeline_copies_estimate") or 1 for item in shared)]
    )
    return {
        "accesses": accesses,
        "grid_blocks": waves["grid_blocks"],
        "sm_count": sm_count,
        "pipeline_depth": pipeline_depth,
    }, unknown


def config_key(config):
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def resolve_oracle(study, family, name, reference, comparison_path):
    """Resolve an archived relative path, falling back to its sealed hash."""
    direct = (comparison_path.parent / reference["path"]).resolve()
    if direct.is_file() and hashlib.sha256(direct.read_bytes()).hexdigest() == reference["sha256"]:
        return direct
    pattern = f"baselines/blackwell/{family}/*/collection/blackwell/test/{name}/brute_force/outcomes.json"
    matches = [
        path
        for path in study.parent.glob(pattern)
        if hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"]
    ]
    if len(matches) != 1:
        raise ValueError(f"{name}: expected one sealed oracle matching {reference['sha256']}, found {len(matches)}")
    return matches[0].resolve()


def case_inputs(study, run_name):
    family = run_name.split("-periodic", 1)[0]
    comparison_path = study / "runs" / run_name / "123" / "blackwell" / "comparison.json"
    comparison = read(comparison_path)
    by_name = {row["workload"]["name"]: row for row in comparison["results"]}
    report_root = comparison_path.parent / "blackwell" / "test"
    for report_path in sorted(report_root.glob("*/tiletune/tiletune.json")):
        name = report_path.parents[1].name
        if name not in by_name:
            raise ValueError(f"{name}: missing from {comparison_path}")
        oracle_ref = by_name[name]["oracle"]
        oracle_path = resolve_oracle(study, family, name, oracle_ref, comparison_path)
        yield name, report_path, oracle_path, oracle_ref["sha256"]


def replay_case(family, name, report_path, oracle_path, oracle_sha256):
    raw = report_path.read_bytes()
    report_sha256 = hashlib.sha256(raw).hexdigest()
    report = json.loads(raw)
    del raw
    sm_count = report["settings"]["device_limits"]["sm_count"]
    records = []
    started = time.perf_counter()
    for record in report["configs"]:
        inputs, unknown = memory_inputs(record, sm_count)
        score = score_memory(**inputs)
        if unknown:
            score.update(score=None, tie_break_score=None, precision="unknown")
            score["unknown"].extend(unknown)
        records.append(
            {
                "index": record["index"],
                "config": record["config"],
                "pre_lowering": record["pre_lowering"],
                "tile_cost": {**score, "ranking_metric": "memory"},
            }
        )
    ranking = rank_records(records)
    scoring_seconds = time.perf_counter() - started

    oracle_raw = oracle_path.read_bytes()
    if hashlib.sha256(oracle_raw).hexdigest() != oracle_sha256:
        raise ValueError(f"oracle changed: {oracle_path}")
    outcomes = json.loads(oracle_raw)
    winner = min((row for row in outcomes if row["status"] == "benchmarked"), key=lambda row: row["latency_ms"])
    del outcomes, oracle_raw
    record_by_config = {config_key(record["config"]): record for record in records}
    winner_record = record_by_config.get(config_key(winner["config"]))
    if winner_record is None:
        raise ValueError(f"{name}: oracle winner is absent from the frozen TileTune pool")
    rank = next(row for row in ranking if row["index"] == winner_record["index"])
    n = len(records)
    return {
        "family": family,
        "workload": name,
        "pool_size": n,
        "scored_count": sum(row["tile_cost"]["score"] is not None for row in records),
        "oracle_index": winner_record["index"],
        "oracle_latency_ms": winner["latency_ms"],
        "oracle_position": rank["position"] if rank["tier"] == "eligible" else None,
        "oracle_rank": rank["rank"] if rank["tier"] == "eligible" else None,
        "oracle_tie_first_rank": rank["tie_first_rank"] if rank["tier"] == "eligible" else None,
        "oracle_tier": rank["tier"],
        "oracle_rank_fraction": rank["rank"] / n if rank["tier"] == "eligible" else None,
        "hits_20_percent": rank["tier"] == "eligible" and rank["rank"] <= (n + 4) // 5,
        "hits_50_percent": rank["tier"] == "eligible" and rank["rank"] <= (n + 1) // 2,
        "scoring_seconds": scoring_seconds,
        "report": {"path": str(report_path), "sha256": report_sha256},
        "oracle": {"path": str(oracle_path), "sha256": oracle_sha256},
    }


def replay(study, output, runs):
    study = Path(study).resolve()
    rows = []
    seen = set()
    for run_name in runs:
        family = run_name.split("-periodic", 1)[0]
        for name, report_path, oracle_path, oracle_sha256 in case_inputs(study, run_name):
            if name in seen:
                raise ValueError(f"duplicate workload from selected runs: {name}")
            seen.add(name)
            row = replay_case(family, name, report_path, oracle_path, oracle_sha256)
            rows.append(row)
            rank = row["oracle_rank"] or "unscored"
            print(f"{name}: oracle tail rank {rank}/{row['pool_size']}", flush=True)
    rows.sort(key=lambda row: (row["family"], row["workload"]))
    result = {
        "version": 1,
        "study": str(study),
        "runs": list(runs),
        "methodology": (
            "Offline replay of frozen TileTune operation facts. Scores are computed before oracle files are opened. "
            "Equal primary scores receive their group's tail rank; no compilation or GPU measurement is performed."
        ),
        "rows": rows,
        "all_oracles_scored": all(row["oracle_rank"] is not None for row in rows),
        "hits_20_percent": sum(row["hits_20_percent"] for row in rows),
        "hits_50_percent": sum(row["hits_50_percent"] for row in rows),
        "all_hit_whole_pool_percent": next(
            (
                percent
                for percent in range(1, 101)
                if all(
                    row["oracle_rank"] is not None
                    and row["oracle_rank"] <= (percent * row["pool_size"] + 99) // 100
                    for row in rows
                )
            ),
            None,
        ),
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output / "report.md").write_text(render(result))
    return result


def render(result):
    rows = result["rows"]
    lines = [
        "# Memory-only TileTune replay",
        "",
        result["methodology"],
        "",
        f"Oracle hits: **{result['hits_20_percent']}/{len(rows)} at 20%**, "
        f"**{result['hits_50_percent']}/{len(rows)} at 50%**.",
        "",
        f"All-case cutoff: **{result['all_hit_whole_pool_percent']}%**, with per-pool budgets rounded up.",
        "",
        "| Family / case | Pool | Scored | Oracle position | Conservative tail rank | Pool share | Replay score ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    families = defaultdict(list)
    for row in rows:
        families[row["family"]].append(row)
        share = f"{100 * row['oracle_rank_fraction']:.2f}%" if row["oracle_rank_fraction"] is not None else "N/A"
        lines.append(
            f"| {row['family']} / `{row['workload']}` | {row['pool_size']} | {row['scored_count']} | "
            f"{row['oracle_position'] or 'Unscored'} | {row['oracle_rank'] or 'Unscored'} | {share} | "
            f"{1000 * row['scoring_seconds']:.2f} |"
        )
    lines += ["", "| Family | Within 20% | Within 50% | Worst tail-rank share |", "|---|---:|---:|---:|"]
    for family, subset in families.items():
        fractions = [row["oracle_rank_fraction"] for row in subset if row["oracle_rank_fraction"] is not None]
        worst = f"{100 * max(fractions):.2f}%" if len(fractions) == len(subset) else "Unscored"
        lines.append(
            f"| {family} | {sum(row['hits_20_percent'] for row in subset)}/{len(subset)} | "
            f"{sum(row['hits_50_percent'] for row in subset)}/{len(subset)} | {worst} |"
        )
    lines += [
        "",
        "This is a fixed-pool retrospective result, not a generalization claim. Memory-event and original-index keys only order "
        "the report within equal primary scores; they do not split a tie for pruning.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories under STUDY/runs, in the desired fixed set")
    args = parser.parse_args()
    result = replay(args.study, args.output, args.runs)
    print(f"Completed {args.output / 'report.md'}: {result['hits_50_percent']}/{len(result['rows'])} within 50%")


if __name__ == "__main__":
    main()
