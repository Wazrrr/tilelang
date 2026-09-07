"""Comparable ranking metrics, independent of measurements and rejection."""

from .pipeline import estimate_pipeline_cycles
from .cta_work import estimate_grid_cycles


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


def rank_records(records):
    """Return all original indices in score order, without reading measurements."""
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
        tier = "pressure_rejected" if decision.get("would_reject") else "unknown" if score is None else "eligible"
        entries.append({"index": record["index"], "tier": tier, "score": score})
    order = {"eligible": 0, "unknown": 1, "pressure_rejected": 2}
    entries.sort(key=lambda e: (order[e["tier"]], e["score"] if e["score"] is not None else float("inf"), e["index"]))
    for i, entry in enumerate(entries):
        entry["rank"] = i + 1
    groups = {}
    for entry in entries:
        groups.setdefault((entry["tier"], entry["score"]), []).append(entry["rank"])
    for entry in entries:
        ranks = groups[entry["tier"], entry["score"]]
        entry.update(tie_first_rank=min(ranks), tie_last_rank=max(ranks))
    return entries
