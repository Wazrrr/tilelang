"""Common module orchestration for every kernel specialization."""

from dataclasses import dataclass
from .specializations import KernelSpecialization, select_specialization


@dataclass
class AnalysisContext:
    func: object
    collector: object
    tile_propagation: object
    config: object
    pass_configs: object = None


def run_modules(context, pressure, device_limits=None):
    from . import cost
    from .ranking import apply_ranking_metric
    from .register_policy import analyze_register_policy

    try:
        specialization = select_specialization(context.collector, context.config.specialization)
    except Exception as error:
        specialization = KernelSpecialization(matched=False, evidence=[f"kernel graph recognition is unresolved: {error}"])
    modules = {}
    try:
        pressure = specialization.register_pressure(context, pressure)
    except Exception as error:
        pressure["tile_liveness"] = {"precision": "unknown", "analysis_error": str(error)}
    modules["register_pressure"] = pressure
    try:
        ws = specialization.warp_specialization(context, pressure)
    except Exception as error:
        ws = {"status": "unknown", "applies": None, "evidence": [str(error)]}
    pressure["warp_specialization"] = ws
    pressure.update(analyze_register_policy(pressure, context.config, specialization, device_limits))
    modules["warp_specialization"] = ws
    tile_cost = {"score": None, "precision": "disabled"}
    modules.update({name: {"precision": "disabled"} for name in ("memory_traffic", "waves", "pipeline_overlap", "ranking")})
    if context.config.ranking:
        try:
            tile_cost = cost.analyze_tile_cost(
                context.collector, context.tile_propagation, pressure, device_limits, context, specialization
            )
            memory, waves = tile_cost.pop("_memory"), tile_cost.pop("_waves")
            modules.update(memory_traffic=memory, waves=waves)
            try:
                pipeline = specialization.pipeline_overlap(context, memory, pressure)
            except Exception as error:
                pipeline = {"precision": "unknown", "unknown": [str(error)], "timing": None, "timing_status": "unknown"}
            modules["pipeline_overlap"] = pipeline
            ranking = apply_ranking_metric(tile_cost, waves, pipeline, context.config, specialization, pressure["register_demand"])
            modules["ranking"] = ranking
            tile_cost.update(score=ranking["score"], score_formula=ranking["formula"], ranking_metric=ranking["metric"])
            if ranking["score"] is None:
                tile_cost["precision"] = "unknown"
                tile_cost["unknown"] = sorted(set(tile_cost.get("unknown", []) + ranking["unknown"]))
        except Exception as error:
            tile_cost = {"score": None, "precision": "unknown", "analysis_error": str(error)}
            for name in ("memory_traffic", "waves", "pipeline_overlap", "ranking"):
                if modules[name].get("precision") == "disabled":
                    modules[name] = {"precision": "unknown", "analysis_error": str(error)}
    for name, result in modules.items():
        # A module exposes the chosen implementation even on its unknown path.
        result.setdefault("implementation", specialization.name if name != "waves" else "generic")
    return {"specialization": specialization.to_dict(), "modules": modules, "pressure": pressure, "tile_cost": tile_cost}
