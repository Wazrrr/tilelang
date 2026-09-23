"""Replay simple memory-score formulas on saved E2 TileTune facts.

Candidate scores are frozen before the E2 oracle indices are loaded.  The
oracle labels are used only to evaluate conservative equal-score tail ranks.
This is a retrospective formula search, not an out-of-sample validation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEPTH_COUNT = (1 << 16) - 1


def ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def encode_bytes_depth_accesses(byte_waves: int, depth: int, access_waves: int) -> int:
    """Exactly encode the lexicographic key (byte_waves, -depth, accesses)."""
    return DEPTH_COUNT * byte_waves * (byte_waves + 1) // 2 + (DEPTH_COUNT - depth) * (byte_waves + 1) + access_waves


def encode_bytes_accesses_depth(byte_waves: int, depth: int, access_waves: int) -> int:
    """Exactly encode the previous H200 key (byte_waves, accesses, -depth)."""
    order = byte_waves * (byte_waves + 1) // 2 + access_waves
    return order * (1 << 16) + (DEPTH_COUNT - depth)


def adjusted_byte_waves(
    byte_waves: int,
    *,
    grid_blocks: int,
    accesses_per_cta: int,
    sm_count: int,
    target_waves: int,
) -> int:
    """Apply an integer launch-underfill correction to logical byte waves.

    A launch below ``target_waves * sm_count`` receives a penalty.  The
    correction is damped by logical accesses per CTA so a long-running CTA is
    not treated like a very small CTA solely because both launch one block.

      shortfall = max(0, target_waves * sm_count - grid_blocks)
      adjusted = ceil(byte_waves * (grid_blocks + accesses_per_cta + shortfall)
                      / (grid_blocks + accesses_per_cta))

    Every input is an IR-derived integer or an explicit device limit.  No
    latency, oracle rank, workload name, or kernel-family rule enters a score.
    """
    denominator = grid_blocks + accesses_per_cta
    shortfall = max(0, target_waves * sm_count - grid_blocks)
    return ceil_div(byte_waves * (denominator + shortfall), denominator)


def grid_only_adjusted_byte_waves(
    byte_waves: int,
    *,
    grid_blocks: int,
    sm_count: int,
    target_waves: int,
) -> int:
    """Apply the same underfill correction using grid coverage alone.

      adjusted = ceil(byte_waves * max(1, target_waves * sm_count / grid_blocks))

    Unlike :func:`adjusted_byte_waves`, this formula does not treat per-CTA
    logical access count as interchangeable with launch blocks.
    """
    return ceil_div(byte_waves * max(grid_blocks, target_waves * sm_count), grid_blocks)


def explicit_rank_key(byte_score: int, grid_blocks: int, accesses_per_cta: int, depth: int, access_waves: int) -> tuple:
    """Return the directly inspectable order (bytes, -grid, -accesses, -depth, access-waves)."""
    return byte_score, -grid_blocks, -accesses_per_cta, -depth, access_waves


def grid_rank_key(byte_score: int, grid_blocks: int, depth: int, access_waves: int) -> tuple:
    """Add grid size as an optional tie-break after adjusted byte waves."""
    return byte_score, -grid_blocks, -depth, access_waves


def candidate_scores(report: dict, sm_count: int, target_waves: range) -> dict[str, dict[int, object]]:
    """Score one workload without receiving its oracle label."""
    scores: dict[str, dict[int, int]] = {
        "h200_current": {},
        "b200_order": {},
        "raw_explicit_order": {},
        "adjusted_explicit_order": {},
        "adjusted_grid_order": {},
        **{f"underfill_{waves}": {} for waves in target_waves},
        **{f"grid_only_{waves}": {} for waves in target_waves},
    }
    seen: set[int] = set()
    for record in report["configs"]:
        index = record["index"]
        if index in seen:
            raise ValueError(f"duplicate candidate index {index}")
        seen.add(index)
        cost = record["tile_cost"]
        module_score = record["modules"]["ranking"]
        byte_waves = cost["logical_byte_waves"]
        access_waves = cost["logical_memory_access_waves"]
        depth = cost["pipeline_depth"]
        grid_blocks = cost["grid_blocks"]
        accesses_per_cta = module_score["logical_memory_accesses_per_cta"]
        waves = ceil_div(grid_blocks, sm_count)
        if access_waves != accesses_per_cta * waves:
            raise ValueError(f"candidate {index} has inconsistent access-wave facts")
        if not (0 <= access_waves <= byte_waves):
            raise ValueError(f"candidate {index} violates 0 <= access_waves <= byte_waves")
        if not (1 <= depth <= DEPTH_COUNT):
            raise ValueError(f"candidate {index} has invalid pipeline depth {depth}")

        scores["h200_current"][index] = encode_bytes_accesses_depth(byte_waves, depth, access_waves)
        scores["b200_order"][index] = encode_bytes_depth_accesses(byte_waves, depth, access_waves)
        scores["raw_explicit_order"][index] = explicit_rank_key(byte_waves, grid_blocks, accesses_per_cta, depth, access_waves)
        for target in target_waves:
            effective_bytes = adjusted_byte_waves(
                byte_waves,
                grid_blocks=grid_blocks,
                accesses_per_cta=accesses_per_cta,
                sm_count=sm_count,
                target_waves=target,
            )
            scores[f"underfill_{target}"][index] = encode_bytes_depth_accesses(effective_bytes, depth, access_waves)
            if target == 3:
                scores["adjusted_explicit_order"][index] = explicit_rank_key(
                    effective_bytes, grid_blocks, accesses_per_cta, depth, access_waves
                )
                scores["adjusted_grid_order"][index] = grid_rank_key(effective_bytes, grid_blocks, depth, access_waves)
            grid_only_bytes = grid_only_adjusted_byte_waves(
                byte_waves,
                grid_blocks=grid_blocks,
                sm_count=sm_count,
                target_waves=target,
            )
            scores[f"grid_only_{target}"][index] = encode_bytes_depth_accesses(grid_only_bytes, depth, access_waves)
    if len(scores["h200_current"]) != len(report["configs"]):
        raise ValueError("candidate count changed while scoring")
    return scores


def tail_ranks(scores: dict[int, object]) -> dict[int, int]:
    """Assign every equal primary score the last position of its group."""
    counts: dict[int, int] = {}
    for score in scores.values():
        counts[score] = counts.get(score, 0) + 1
    tails: dict[int, int] = {}
    position = 0
    for score in sorted(counts):
        position += counts[score]
        tails[score] = position
    return {index: tails[score] for index, score in scores.items()}


def freeze_rankings(analysis_root: Path, workload_names: list[str], sm_count: int, targets: range) -> dict:
    """Load candidate facts and finish every formula ranking before labels."""
    frozen = {}
    for name in workload_names:
        report_path = analysis_root / name / "tiletune.json"
        report = json.loads(report_path.read_text())
        scores = candidate_scores(report, sm_count, targets)
        frozen[name] = {
            "pool_size": len(report["configs"]),
            "ranks": {formula: tail_ranks(values) for formula, values in scores.items()},
        }
    return frozen


def evaluate(frozen: dict, oracle_summary: dict, targets: range) -> dict:
    labels = {row["workload"]: [oracle["index"] for oracle in row["oracles"]] for row in oracle_summary["workloads"]}
    if set(labels) != set(frozen):
        raise ValueError("workload names differ between frozen rankings and oracle summary")
    formulas = [
        "h200_current",
        "b200_order",
        "raw_explicit_order",
        "adjusted_explicit_order",
        "adjusted_grid_order",
        *[f"underfill_{target}" for target in targets],
        *[f"grid_only_{target}" for target in targets],
    ]
    rows = []
    for name, item in frozen.items():
        pool_size = item["pool_size"]
        ranks = {}
        for formula in formulas:
            try:
                rank = max(item["ranks"][formula][index] for index in labels[name])
            except KeyError as error:
                raise ValueError(f"oracle index {error.args[0]} is absent from {name}") from error
            ranks[formula] = {
                "tail_rank": rank,
                "pool_fraction": rank / pool_size,
                "within_50_percent": rank <= math.floor(0.5 * pool_size),
            }
        rows.append({"workload": name, "pool_size": pool_size, "oracle_indices": labels[name], "ranks": ranks})

    aggregate = {}
    for formula in formulas:
        values = [row["ranks"][formula] for row in rows]
        worst_index = max(range(len(rows)), key=lambda index: values[index]["pool_fraction"])
        aggregate[formula] = {
            "within_50_percent": sum(value["within_50_percent"] for value in values),
            "workload_count": len(values),
            "worst_workload": rows[worst_index]["workload"],
            "worst_tail_rank": values[worst_index]["tail_rank"],
            "worst_pool_size": rows[worst_index]["pool_size"],
            "worst_pool_fraction": values[worst_index]["pool_fraction"],
        }
    return {"aggregate": aggregate, "workloads": rows}


def write_report(result: dict, output: Path) -> None:
    aggregate = result["aggregate"]
    chosen = aggregate["underfill_3"]
    grid_only = aggregate["grid_only_3"]
    explicit = aggregate["adjusted_explicit_order"]
    grid_order = aggregate["adjusted_grid_order"]
    lines = [
        "# E2 memory-formula replay",
        "",
        "Candidate rankings were frozen from saved CPU-only TileTune facts before E2 oracle indices were loaded. "
        "Equal scores use conservative tail ranks.",
        "",
        "## Formula comparison",
        "",
        "| Formula | Within 50% | Worst workload | Worst tail rank | Worst share |",
        "|---|---:|---|---:|---:|",
    ]
    display = {
        "h200_current": "previous H200 `(B, E, -D)`",
        "b200_order": "B200 `(B, -D, E)`",
        "raw_explicit_order": "raw explicit `(B, -G, -e, -D, E)`",
        "adjusted_explicit_order": "adjusted explicit `(U, -G, -e, -D, E)`",
        "adjusted_grid_order": "adjusted with grid tie-break `(U, -G, -D, E)`",
        **{f"underfill_{target}": f"access-damped underfill target={target}" for target in range(1, 9)},
        **{f"grid_only_{target}": f"grid-only underfill target={target}" for target in range(1, 9)},
    }
    for formula, item in aggregate.items():
        lines.append(
            f"| {display[formula]} | {item['within_50_percent']}/{item['workload_count']} | "
            f"{item['worst_workload']} | {item['worst_tail_rank']}/{item['worst_pool_size']} | "
            f"{100 * item['worst_pool_fraction']:.2f}% |"
        )
    lines += [
        "",
        "## Selected simple candidate",
        "",
        "For `B = logical_byte_waves`, `E = logical_access_waves`, `e = logical accesses per CTA`, "
        "`G = grid_blocks`, `S = SM_count`, and `D = pipeline_depth`:",
        "",
        "```text",
        "shortfall = max(0, 3*S - G)",
        "U = ceil(B * (G + e + shortfall) / (G + e))",
        "rank key = (U, -D, E)",
        "```",
        "",
        f"This retrospective fit keeps {chosen['within_50_percent']}/{chosen['workload_count']} E2 oracles within 50%. "
        f"The worst case is `{chosen['worst_workload']}` at "
        f"{chosen['worst_tail_rank']}/{chosen['worst_pool_size']} ({100 * chosen['worst_pool_fraction']:.2f}%).",
        "",
        "## Removing per-CTA access count",
        "",
        "The clearer grid-only correction is:",
        "",
        "```text",
        "shortfall = max(0, 3*S - G)",
        "U_grid = ceil(B * (1 + shortfall/G))",
        "       = ceil(B * max(1, 3*S/G))",
        "rank key = (U_grid, -D, E)",
        "```",
        "",
        f"It keeps {grid_only['within_50_percent']}/{grid_only['workload_count']} E2 oracles within 50%. "
        f"The worst case is `{grid_only['worst_workload']}` at "
        f"{grid_only['worst_tail_rank']}/{grid_only['worst_pool_size']} "
        f"({100 * grid_only['worst_pool_fraction']:.2f}%).",
        "",
        "## Explicit rank order",
        "",
        "The scalar encoding is unimportant when it preserves the same lexicographic order. The adjusted first term "
        "is still essential: raw `B` first reaches only 23/25, while the following order reaches "
        f"{explicit['within_50_percent']}/{explicit['workload_count']}:",
        "",
        "```text",
        "U = ceil(B * (G + e + max(0, 3*S - G)) / (G + e))",
        "rank key = (U, -G, -e, -D, E)",
        "```",
        "",
        "Because `E = e * ceil(G/S)`, `E` is redundant after both `G` and `e`. Moreover, `e` already affects `U`; "
        "using `-e` again double-rewards high access count. Adding only `-G` gives:",
        "",
        "```text",
        "rank key = (U, -G, -D, E)",
        "```",
        "",
        f"It retains {grid_order['within_50_percent']}/{grid_order['workload_count']} workloads; its worst case is "
        f"`{grid_order['worst_workload']}` at {grid_order['worst_tail_rank']}/{grid_order['worst_pool_size']} "
        f"({100 * grid_order['worst_pool_fraction']:.2f}%). Because this does not improve 50% retention over "
        "the simpler `(U, -D, E)` key, the production formula does not include `-G`.",
        "",
        "| Workload | Pool | Previous | Selected `(U, -D, E)` | Full explicit | Extra grid | Selected share | Grid-only |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["workloads"]:
        current = row["ranks"]["h200_current"]["tail_rank"]
        selected = row["ranks"]["underfill_3"]
        explicit_order = row["ranks"]["adjusted_explicit_order"]
        grid_order = row["ranks"]["adjusted_grid_order"]
        grid = row["ranks"]["grid_only_3"]
        lines.append(
            f"| {row['workload']} | {row['pool_size']} | {current} | {selected['tail_rank']} | "
            f"{explicit_order['tail_rank']} | {grid_order['tail_rank']} | {100 * selected['pool_fraction']:.2f}% | "
            f"{grid['tail_rank']} |"
        )
    lines += [
        "",
        "The three-wave target was selected using these same E2 oracle labels. This establishes fixed-pool fit only; "
        "it is not evidence of generalization and is not a GPU performance measurement.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.analysis_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    workload_names = [row["workload"] for row in summary["workloads"]]
    sm_count = summary["device_limits"]["sm_count"]
    targets = range(1, 9)

    # Do not pass labels into this stage. All candidate rankings are complete
    # before the oracle-bearing summary is joined below.
    frozen = freeze_rankings(args.analysis_root, workload_names, sm_count, targets)
    result = evaluate(frozen, summary, targets)
    result.update(
        method="CPU-only replay of saved E2 TileTune facts",
        analysis_root=str(args.analysis_root.resolve()),
        sm_count=sm_count,
        selected_formula="underfill_3",
        tested_grid_only_formula="grid_only_3",
        fitted_on_same_oracles=True,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_report(result, args.output)
    selected = result["aggregate"]["underfill_3"]
    print(
        f"selected order: {selected['within_50_percent']}/{selected['workload_count']} within 50%; "
        f"worst {selected['worst_workload']} "
        f"{selected['worst_tail_rank']}/{selected['worst_pool_size']}"
    )


if __name__ == "__main__":
    main()
