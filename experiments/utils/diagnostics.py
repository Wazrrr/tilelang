"""Evaluate frozen rankings against a separate correctness-checked oracle table."""

from collections import Counter, defaultdict
import math
import random
import statistics

from experiments.utils.results import oracle_at_k


def _ranks(values):
    positions = defaultdict(list)
    for rank, index in enumerate(sorted(range(len(values)), key=values.__getitem__), 1):
        positions[values[index]].append(rank)
    return [statistics.mean(positions[value]) for value in values]


def spearman(left, right):
    if len(left) < 3 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    x, y = _ranks(left), _ranks(right)
    xm, ym = statistics.mean(x), statistics.mean(y)
    return sum((a - xm) * (b - ym) for a, b in zip(x, y)) / math.sqrt(sum((a - xm) ** 2 for a in x) * sum((b - ym) ** 2 for b in y))


def assess(report, oracle_records, *, seed=123):
    times = {r["index"]: r["latency_ms"] for r in oracle_records if r["status"] == "benchmarked"}
    ranking = report.get("ranking", [])
    records = report.get("configs", [])
    finite_scores = [r for r in ranking if type(r.get("score")) in (int, float) and math.isfinite(r["score"])]
    eligible = [r for r in finite_scores if r.get("tier") == "eligible"]
    paired = [r for r in eligible if r["index"] in times]
    selected = (report.get("selection") or {}).get("selected_indices", [])
    best = min(times, key=times.get) if times else None

    def retained(indices):
        return oracle_at_k(times, indices)

    curves = {}
    rng = random.Random(seed)
    for k in sorted({1, 5, 10, 20, 50, len(selected)} - {0}):
        random_values = [retained(rng.sample(range(len(records)), min(k, len(records)))) for _ in range(500)] if records else []
        finite = [v for v in random_values if v is not None]
        curves[str(k)] = dict(
            oracle_at_k=retained([r["index"] for r in eligible[:k]]),
            selected_count=min(k, len(eligible)),
            random_mean_oracle_at_k=statistics.mean(finite) if finite else None,
            random_no_success_fraction=(len(random_values) - len(finite)) / len(random_values) if random_values else None,
        )
    unknown = Counter()
    by_stage = defaultdict(list)
    relative_errors = []
    latency_ratios = []
    for record in records:
        modules = record.get("modules") or {}
        for reason in (modules.get("pipeline_overlap") or {}).get("unknown", []):
            unknown[reason] += 1
        index = record["index"]
        if index in times:
            cfg = record["config"]
            stage = cfg.get("stages", cfg.get("num_stages"))
            if stage is not None:
                by_stage[str(stage)].append(times[index])
            predicted = (modules.get("ranking") or {}).get("estimated_latency_ms")
            if predicted is not None and predicted > 0:
                relative_errors.append(abs(predicted / times[index] - 1))
                latency_ratios.append(predicted / times[index])
    best_rank = next((r for r in ranking if r["index"] == best), None)
    return dict(
        grid_size=len(records),
        oracle_successes=len(times),
        eligible_scores=len(eligible),
        paired_scores=len(paired),
        correct_score_coverage=sum(r["index"] in times for r in finite_scores) / len(times) if times else None,
        score_coverage=len(eligible) / len(records) if records else 0,
        unique_eligible_scores=len({r["score"] for r in eligible}),
        spearman_score_latency=spearman([r["score"] for r in paired], [times[r["index"]] for r in paired]),
        oracle_winner_index=best,
        oracle_winner_rank=best_rank,
        oracle_best_ms=times.get(best),
        selected_count=len(selected),
        selected_oracle_successes=sum(i in times for i in selected),
        oracle_at_k=retained(selected),
        curves=curves,
        ranking_tiers=dict(Counter(r.get("tier", "unknown") for r in ranking)),
        pipeline_unknown_reasons=dict(unknown),
        best_ms_by_stages={key: min(values) for key, values in by_stage.items()},
        predicted_latency_pairs=len(relative_errors),
        median_absolute_relative_prediction_error=statistics.median(relative_errors) if relative_errors else None,
        median_predicted_over_measured=statistics.median(latency_ratios) if latency_ratios else None,
    )
