"""Re-score frozen TileTune operation facts without rebuilding or benchmarking.

python -m experiments.replay_memory --study /path/to/completed/study --output /path/to/new/report

The scorer sees only memory facts. Oracle labels are joined after the complete
ranking has been written. This is a retrospective result on the supplied pool.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from math import prod
from pathlib import Path
import time

from experiments.compare_results import compare
from experiments.utils.results import provenance, read
from tiletune_core.memory import score_memory
from tiletune_core.ranking import rank_records


def literal_int(value):
    """Archived expressions are not executable code; accept integer literals only."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def memory_inputs(record, sm_count):
    """Export the same logical accesses as the live IR adapter from saved facts.

    Symbolic loop bounds (e.g. causal attention) use the archived per-operation
    visit upper bound. No timing score, latency, configuration knob or family
    name is used to derive memory work.
    """
    storage = {}
    for item in record["pressure"]["logical_storage"]:
        if item["scope"] != "global":
            continue
        if item["buffer"] in storage:
            raise ValueError("archived global buffer names must be unique to resolve dtype widths")
        storage[item["buffer"]] = item
    memory = record["modules"]["memory_traffic"]
    accesses, unknown = [], []
    for op in record["tile_propagation"]["operations"]:
        if op["unknown"]:
            unknown.append(f"operation {op['index']}: unresolved memory effects")
        external = [(direction, region) for direction in ("reads", "writes") for region in op[direction] if region["scope"] == "global"]
        if not external:
            continue
        extents = [literal_int(loop["extent"]) for loop in op["loops"] if loop["kind"] != "4"]
        if all(v is not None and v >= 0 for v in extents):
            visits = prod(extents)
        else:
            bounds = {item["visits_per_block"] for item in memory["input_tiles"] if item.get("operation") == op["index"]}
            if len(bounds) != 1 or None in bounds:
                raise ValueError(f"operation {op['index']}: no unambiguous archived loop-visit bound")
            visits = bounds.pop()
        for direction, region in external:
            item = storage[region["buffer"]]
            bits, elements = item["logical_bits"], item["logical_elements"]
            if bits is None or not elements or bits % elements:
                raise ValueError("archived storage does not resolve dtype width")
            extents = [literal_int(axis["extent"]) for axis in region["ranges"]]
            volume = prod(extents) if all(v is not None and v >= 0 for v in extents) else None
            accesses.append(
                dict(
                    operation=op["index"],
                    direction=direction,
                    buffer=region["buffer"],
                    bytes=(volume * (bits // elements) + 7) // 8 if volume is not None else None,
                    visits=visits,
                )
            )
    return dict(accesses=accesses, grid_blocks=record["modules"]["waves"]["grid_blocks"], sm_count=sm_count), unknown


def replay(study, output):
    study, output = Path(study).resolve(), Path(output).resolve()
    if output == study:
        raise ValueError("output must be separate from the frozen study")
    source = read(study / "comparison.json")
    paths = sorted((study / "comparison").glob("*/*/oracle-curves.json"))
    if source["status"] != "completed" or len(paths) != source.get("expected_shapes", source.get("expected_comparisons")):
        raise ValueError("a completed study with every oracle comparison is required")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in paths:
        saved = read(path)
        ref = next(m["input"] for m in saved["methods"] if m["method"] == "tiletune")
        raw = Path(ref["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
            raise ValueError(f"frozen TileTune report changed: {ref['path']}")
        report = json.loads(raw)
        del raw
        case = read(path.parent / "comparison.json")
        name = case["workload"]["name"]
        sm_count = report["settings"]["device_limits"]["sm_count"]
        records = []
        started = time.perf_counter()
        for record in report["configs"]:
            inputs, unknown = memory_inputs(record, sm_count)
            ranking = score_memory(**inputs)
            if unknown:
                ranking.update(score=None, tie_break_score=None, precision="unknown")
                ranking["unknown"].extend(unknown)
            records.append(
                dict(
                    index=record["index"],
                    config=record["config"],
                    status="analyzed",
                    pre_lowering=record["pre_lowering"],
                    memory_inputs=inputs,
                    tile_cost=dict(ranking, ranking_metric="memory"),
                )
            )
        ranking = rank_records(records)
        scoring_seconds = time.perf_counter() - started
        target = output / path.parent.parent.name / name
        target.mkdir(parents=True, exist_ok=True)
        ranked_path = target / "tiletune.json"
        ranked_path.write_text(
            json.dumps(
                dict(
                    version=1,
                    settings=dict(ranking_metric="memory", sm_count=sm_count),
                    measurement_scope="ranking only; no new compilation or GPU measurements",
                    source=ref,
                    scoring_seconds=scoring_seconds,
                    configs=records,
                    ranking=ranking,
                ),
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        # Preserve workload/target identity for the common-oracle join.
        original_manifest = Path(ref["path"]).parent / "experiment.json"
        if original_manifest.is_file():
            (target / "experiment.json").write_bytes(original_manifest.read_bytes())
        n = len(records)
        ks = sorted({20, 100, (n + 4) // 5, (n + 1) // 2, n})
        comparison = compare(saved["oracle"]["sources"][0]["path"], {"tiletune": ranked_path}, ks)
        if comparison["oracle"]["sources"] != saved["oracle"]["sources"]:
            raise ValueError("oracle changed since the original study")
        (target / "oracle-curves.json").write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n")
        method = comparison["methods"][0]
        first = method["first_oracle_hit_k"]
        original_records = {r["index"]: r for r in report["configs"]}
        rows.append(
            dict(
                workload=name,
                family=case.get("family", path.parent.parent.name),
                seed=case.get("seed", source.get("protocol", {}).get("seed")),
                pool_size=n,
                scored_count=sum(r["tile_cost"]["score"] is not None for r in records),
                eligible_count=method["available_count"],
                first_oracle_hit_k=first,
                oracle_rank_fraction=first / n if first is not None else None,
                hits_20_percent=first is not None and first <= (n + 4) // 5,
                hits_50_percent=first is not None and first <= (n + 1) // 2,
                scoring_seconds=scoring_seconds,
                old_first_oracle_hit_k=next(
                    (
                        i + 1
                        for i, r in enumerate(v for v in report["ranking"] if v["tier"] == "eligible")
                        if original_records[r["index"]]["config"] == comparison["oracle"]["best_config"]
                    ),
                    None,
                ),
                source=ref,
                ranking=provenance(ranked_path),
                curves=method["curves"],
            )
        )
        print(f"{name}: oracle rank {first}/{n}; scored {rows[-1]['scored_count']}/{n}", flush=True)
        del report, records
    root = Path(__file__).resolve().parents[1]
    result = dict(
        study=provenance(study / "comparison.json"),
        code=[
            provenance(root / p)
            for p in ("tiletune_core/memory.py", "tiletune_core/ranking.py", "tilelang/tiletune/memory.py", "experiments/replay_memory.py")
        ],
        semantics="Retrospective replay of frozen collector facts. Kernels, pool, input values, oracle timings and hard resource policies are unchanged. No GPU work. Scoring time excludes IR capture and file I/O.",
        rows=rows,
        all_oracles_scored=all(r["first_oracle_hit_k"] is not None for r in rows),
        hits_20_percent=sum(r["hits_20_percent"] for r in rows),
        hits_50_percent=sum(r["hits_50_percent"] for r in rows),
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output / "report.md").write_text(render(result))
    return result


def render(result):
    rows = result["rows"]
    lines = [
        "# Memory-only TileTune replay",
        "",
        result["semantics"],
        "",
        f"Oracle hits: **{result['hits_20_percent']}/{len(rows)} at 20%**, **{result['hits_50_percent']}/{len(rows)} at 50%**.",
        "",
        "Primary score: logical global bytes per CTA × ceil(grid CTAs / SMs). Break equal-byte ties by fewer logical memory accesses per wave, then original index. No compute rates, inferred overlap, or occupancy prediction.",
        "",
        "| Case | Pool | Scored | Previous oracle rank | New oracle rank | Pool share | Replay scoring ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    families = defaultdict(list)
    for row in rows:
        families[row["family"]].append(row)
        first = row["first_oracle_hit_k"]
        share = f"{100 * row['oracle_rank_fraction']:.2f}%" if first is not None else "N/A"
        lines.append(
            f"| {row['workload']} | {row['pool_size']} | {row['scored_count']} | {row['old_first_oracle_hit_k'] or 'Unreachable'} | {first or 'Unreachable'} | {share} | {1000 * row['scoring_seconds']:.2f} |"
        )
    lines += ["", "| Family | Within 20% | Within 50% | Worst oracle rank / pool |", "|---|---:|---:|---:|"]
    for family, subset in families.items():
        worst = max((r["oracle_rank_fraction"] for r in subset if r["oracle_rank_fraction"] is not None), default=None)
        value = f"{100 * worst:.2f}%" if worst is not None and all(r["oracle_rank_fraction"] is not None for r in subset) else "Unreachable"
        lines.append(
            f"| {family} | {sum(r['hits_20_percent'] for r in subset)}/{len(subset)} | {sum(r['hits_50_percent'] for r in subset)}/{len(subset)} | {value} |"
        )
    lines += [
        "",
        "This fixed-pool research result does not establish generalization or fresh GPU speedups. Invalid measured candidates still consume shortlist slots. Unknown memory effects remain explicit; storage/schedule uncertainty alone does not suppress scores.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.study, args.output)
    print(f"Completed {args.output / 'report.md'}: {result['hits_50_percent']}/{len(result['rows'])} within 50%")


if __name__ == "__main__":
    main()
