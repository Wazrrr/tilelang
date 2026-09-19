"""Conservative score ranks and selection that retains boundary ties."""

from .pipeline import estimate_pipeline_cycles
from .schedule import estimate_grid_cycles


def apply_ranking_metric(tile_cost, waves, pipeline, config, specialization, register_demand=None):
    metric = config.ranking_metric
    result = {"metric": metric, "score": None, "precision": "unknown", "formula": None, "unknown": []}
    if not specialization.matched:
        result["unknown"].append("requested specialization did not match the IR")
        return result
    demand = register_demand or {}
    if demand.get("status") == "exceeds_allowance":
        result["unknown"].append(
            "estimated live tile demand exceeds the soft register allowance; physical occupancy is reported separately"
        )
        return result
    if metric == "traffic_waves":
        result.update(score=tile_cost["score"], formula="(traffic_bytes_per_block + 1) * num_waves_estimate", units="byte-waves")
    else:
        result.update(formula="num_waves_estimate * estimated_wave_cycles", units="cycles")
        if not waves["unknown"] and not tile_cost["unknown"]:
            resident = waves["resident_blocks_per_sm_estimate"]
            sms = waves["device_limits"]["sm_count"]
            grid = waves["grid_blocks"]
            active = min(resident, (grid + sms - 1) // sms)
            timing = estimate_pipeline_cycles(pipeline, concurrent_ctas=active)
            if timing is not None:
                scale = (config.performance_model or {}).get("latency_scale", 1)
                grid_timing = estimate_grid_cycles(
                    pipeline.get("cta_work", {}),
                    lambda n: estimate_pipeline_cycles(pipeline, concurrent_ctas=active, iterations=n),
                    sms * active,
                    waves["num_waves_estimate"],
                )
                if grid_timing is None:
                    result["unknown"].append("unresolved CTA work distribution")
                    return result
                result.update(score=grid_timing["cycles"] * scale, wave_timing=timing, grid_timing=grid_timing, latency_scale=scale)
                if grid_timing["method"] != "uniform CTA waves":
                    result["formula"] = "estimated_grid_cycles_from_CTA_work_distribution"
                clock = (config.performance_model or {}).get("reference_clock_mhz")
                if clock:
                    result["estimated_latency_ms"] = result["score"] / (1000 * clock)
                if scale != 1:
                    result["formula"] = "latency_scale * (" + result["formula"] + ")"
        if result["score"] is None:
            result["unknown"].append("pipeline_time requires a supported schedule, complete effective cost profile, and occupancy inputs")
    if result["score"] is not None:
        result["precision"] = "estimate"
        if demand.get("status") == "within_allowance":
            result["conditional_on_spill_allowance"] = True
            result["spill_traffic_modeled"] = False
            result["assumptions"] = demand["assumptions"]
    return result


def alpha_budget(pool_size, alpha):
    """Strict fraction of the original pool, including failed/unknown candidates."""
    import math

    if type(pool_size) is not int or pool_size <= 0:
        raise ValueError("pool_size must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, int | float) or not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    budget = math.floor(pool_size * alpha)
    if budget == 0:
        raise ValueError("alpha selects no candidates from the supplied pool")
    return budget


def select_top_k(ranking, k, *, include_ties=True, strict_budget=False):
    """Keep the first k eligible candidates and their complete boundary tie.

    Equal primary scores are inseparable by default. Fixed-budget historical
    baselines can explicitly request ``include_ties=False``.
    A strict budget excludes the complete crossing group instead of expanding it.
    """
    import math

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not isinstance(strict_budget, bool):
        raise ValueError("strict_budget must be a bool")
    if strict_budget and not include_ties:
        raise ValueError("strict_budget requires include_ties=True")
    eligible = [entry for entry in ranking if entry["tier"] == "eligible" and entry["score"] is not None and math.isfinite(entry["score"])]
    if strict_budget:
        return [entry["index"] for entry in eligible if entry["tie_last_rank"] <= k]
    if not include_ties or len(eligible) <= k:
        return [entry["index"] for entry in eligible[:k]]
    boundary = eligible[k - 1]["score"]
    return [entry["index"] for position, entry in enumerate(eligible) if position < k or entry["score"] == boundary]


def select_with_exploration(ranking, records, k, *, fraction=0.2, seed=123):
    """Freeze ranked and unknown-cost attempts using only declared configurations."""
    import hashlib
    import json
    import math
    from collections import defaultdict, deque

    ranked = select_top_k(ranking, k)
    if isinstance(fraction, bool) or not isinstance(fraction, int | float) or not 0 < fraction <= 1:
        raise ValueError("exploration fraction must be in (0, 1]")
    if type(seed) is not int or seed < 0:
        raise ValueError("exploration seed must be a nonnegative integer")
    by_id = {r["index"]: r for r in records}
    strata = defaultdict(list)
    for entry in ranking:
        record = by_id[entry["index"]]
        decision = record.get("pre_lowering") or (record.get("pressure") or {}).get("decision") or {}
        if (
            entry["tier"] != "unknown"
            or decision.get("would_reject")
            or record.get("status") in ("elaboration_failed", "analysis_failed", "pre_lowering_rejected")
        ):
            continue
        config = record["config"]
        stratum = (
            config.get("implementation", "original"),
            str(config.get("stages", config.get("num_stages", 0))),
            str(config.get("threads", 0)),
        )
        identity = json.dumps(dict(seed=seed, config=config), sort_keys=True, separators=(",", ":"))
        strata[stratum].append((hashlib.sha256(identity.encode()).hexdigest(), entry["index"]))
    queues = [deque(index for _, index in sorted(strata[key])) for key in sorted(strata)]
    pool = []
    while any(queues):
        for queue in queues:
            if queue:
                pool.append(queue.popleft())
    reserved = min(len(pool), math.ceil(k * fraction))
    selected = ranked if not reserved else select_top_k(ranking, k - reserved) if k > reserved else []
    explored = pool[: max(reserved, k - len(selected))]
    return selected + explored, explored


def assign_tail_ranks(entries):
    """Annotate an ordered report with positions and primary-score tie ranges."""
    groups = {}
    for position, entry in enumerate(entries, 1):
        entry["position"] = position
        groups.setdefault((entry["tier"], entry["score"]), []).append(position)
    for entry in entries:
        positions = groups[entry["tier"], entry["score"]]
        entry.update(rank=positions[-1], tie_first_rank=positions[0], tie_last_rank=positions[-1])
    return entries


def rank_records(records):
    """Order candidates and assign equal primary scores their group's tail rank.

    Secondary keys only order the report within a tie. They cannot make an
    equal-score candidate appear safer to prune. ``position`` records that
    deterministic order separately from the conservative predicted ``rank``.
    """
    import math

    if len({r["index"] for r in records}) != len(records):
        raise ValueError("ranking requires unique original indices")
    metrics = {
        (r.get("tile_cost") or {}).get("ranking_metric", "traffic_waves")
        for r in records
        if (r.get("tile_cost") or {}).get("score") is not None
    }
    if len(metrics) > 1:
        raise ValueError("cannot rank scores with different ranking metrics/units together")
    entries = []
    for record in records:
        pressure = record.get("pressure") or {}
        decision = record.get("pre_lowering") or pressure.get("decision") or {}
        score = (record.get("tile_cost") or {}).get("score")
        if score is not None and (type(score) not in (float, int) or not math.isfinite(score)):
            score = None
        tier = "pressure_rejected" if decision.get("would_reject") else "unknown" if score is None else "eligible"
        entry = {"index": record["index"], "tier": tier, "score": score}
        cost = record.get("tile_cost") or {}
        if cost.get("ranking_metric") == "memory":
            secondary = cost.get("tie_break_score")
            if score is not None and (type(secondary) not in (int, float) or not math.isfinite(secondary) or secondary < 0):
                raise ValueError("memory ranking requires a finite nonnegative tie_break_score")
            entry["tie_break_score"] = secondary
        entries.append(entry)
    order = {"eligible": 0, "unknown": 1, "pressure_rejected": 2}
    entries.sort(
        key=lambda e: (
            order[e["tier"]],
            e["score"] if e["score"] is not None else float("inf"),
            e.get("tie_break_score") or 0,
            e["index"],
        )
    )
    return assign_tail_ranks(entries)


def combine_tile_cost(memory, waves):
    traffic, grid = memory["traffic_bytes_per_block"], waves["grid_blocks"]
    unknown = sorted(set(memory["unknown"] + waves["unknown"]))
    count = waves["num_waves_estimate"]
    score = (traffic + 1) * count if not unknown and traffic is not None and count is not None else None
    return {
        **memory,
        **waves,
        "traffic_bytes_grid_estimate": traffic * grid if traffic is not None and grid is not None else None,
        "score": score,
        "score_formula": "(traffic_bytes_per_block + 1) * num_waves_estimate",
        "precision": "unknown" if unknown else "estimate",
        "unknown": unknown,
        "assumptions": memory["assumptions"] + waves["assumptions"] + ["ranking never adds rejections or truncates configs"],
    }
