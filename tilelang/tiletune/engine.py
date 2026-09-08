"""Common analysis flow: recognize, estimate registers, model execution, rank.

Family policy lives in families/. Stages receive a resolved analysis context;
unexpected errors propagate to the caller instead of producing partial reports.
"""

from dataclasses import dataclass

from .budget import resolve_register_budget
from .families import select_specialization


@dataclass
class AnalysisContext:
    func: object
    collector: object
    tile_propagation: object
    config: object
    target: object
    device_limits: object
    pass_configs: dict
    trace: object


def run_modules(context, pressure):
    """Run the common stages and make one final register decision."""
    from . import cost
    from .ranking import apply_ranking_metric
    from .register_policy import analyze_register_policy

    trace = context.trace
    specialization = select_specialization(context.collector, context.config.specialization)
    trace.record("specialization", lambda: specialization.to_dict())

    pressure = specialization.register_pressure(context, pressure)
    trace.record("pressure.tile_liveness", lambda: pressure["tile_liveness"])
    pressure.update(resolve_register_budget(context.config, context.target))
    ws = specialization.warp_specialization(context, pressure)
    pressure["warp_specialization"] = ws
    trace.record("pressure.warp_specialization", lambda: ws)
    pressure.update(analyze_register_policy(pressure, context.config, specialization, context.device_limits))
    trace.record(
        "pressure.register_policy", lambda: {key: pressure[key] for key in ("physical_register_allocation", "register_demand", "decision")}
    )
    modules = {"register_pressure": pressure, "warp_specialization": ws}

    tile_cost = {"score": None, "precision": "disabled"}
    modules.update({name: {"precision": "disabled"} for name in ("memory_traffic", "waves", "pipeline_overlap", "ranking")})
    if context.config.ranking:
        tile_cost = cost.analyze_tile_cost(context, specialization, pressure)
        memory, waves = tile_cost.pop("_memory"), tile_cost.pop("_waves")
        modules.update(memory_traffic=memory, waves=waves)
        trace.record("memory", lambda: memory)
        trace.record("waves", lambda: waves)

        pipeline = specialization.pipeline_overlap(context, memory, pressure)
        modules["pipeline_overlap"] = pipeline
        trace.record("pipeline", lambda: pipeline)
        ranking = apply_ranking_metric(tile_cost, waves, pipeline, context.config, specialization, pressure["register_demand"])
        modules["ranking"] = ranking
        trace.record("ranking", lambda: ranking)
        tile_cost.update(score=ranking["score"], score_formula=ranking["formula"], ranking_metric=ranking["metric"])
        if ranking["score"] is None:
            tile_cost["precision"] = "unknown"
            tile_cost["unknown"] = sorted(set(tile_cost["unknown"] + ranking["unknown"]))
    else:
        trace.record("ranking", lambda: {"precision": "disabled", "reason": "config.ranking is False"})

    for name, result in modules.items():
        result["implementation"] = specialization.name if name != "waves" else "generic"
    return {"specialization": specialization.to_dict(), "modules": modules, "pressure": pressure, "tile_cost": tile_cost}
