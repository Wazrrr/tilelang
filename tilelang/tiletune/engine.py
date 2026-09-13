"""Common analysis flow: recognize, estimate registers, model execution, rank.

Family policy lives in families/. Stages receive a resolved analysis context;
unexpected errors propagate to the caller instead of producing partial reports.
"""

from dataclasses import dataclass

from . import global_memory, shared_memory, occupancy, pipeline, register_pressure
from .register_pressure import resolve_register_budget, analyze_register_policy
from .tile_liveness import analyze_live_tiles
from .warp_specialization import predict_warp_specialization
from .ranking import combine_tile_cost
from .src.collector import _Collector
from .src.propagation import _propagate_tiles, _kernel_outputs
from .src.buffer_facts import collect_buffer_facts
from .trace import AnalysisTrace, collector_snapshot, propagation_snapshot
from .families import select_specialization


@dataclass
class AnalysisContext:
    func: object
    collector: object
    buffer_facts: dict
    tile_propagation: object
    config: object
    target: object
    device_limits: object
    pass_configs: dict
    trace: object


def run_modules(context, pressure):
    """Run the common stages and make one final register decision."""
    from .ranking import apply_ranking_metric

    trace = context.trace
    specialization = select_specialization(context.collector, context.config.specialization)
    trace.record("specialization", lambda: specialization.to_dict())

    phase_labels = {op.index: specialization.phase(op) for op in context.collector.operations}
    pressure["tile_liveness"] = analyze_live_tiles(context.collector, context.buffer_facts, loop=specialization.loop)
    for phase in pressure["tile_liveness"]["phases"]:
        phase["phase"] = phase_labels[phase["operation"]]
    trace.record("pressure.tile_liveness", lambda: pressure["tile_liveness"])
    pressure.update(resolve_register_budget(context.config, context.target))
    ws = predict_warp_specialization(
        context.func, context.collector, pressure, context.pass_configs, policy=specialization.warp_specialization_policy()
    )
    pressure["warp_specialization"] = ws
    trace.record("pressure.warp_specialization", lambda: ws)
    pressure.update(
        analyze_register_policy(
            pressure, context.config, context.device_limits, spill_allowance=specialization.register_spill_allowance(context.config)
        )
    )
    trace.record(
        "pressure.register_policy", lambda: {key: pressure[key] for key in ("physical_register_allocation", "register_demand", "decision")}
    )
    modules = {"register_pressure": pressure, "warp_specialization": ws}

    tile_cost = {"score": None, "precision": "disabled"}
    modules.update({name: {"precision": "disabled"} for name in ("memory_traffic", "waves", "pipeline_overlap", "ranking")})
    if context.config.ranking:
        memory = global_memory.analyze_global_memory(
            context.collector,
            context.tile_propagation,
            context.buffer_facts,
            loop=specialization.loop,
            actual_accesses=specialization.actual_memory_accesses,
        )
        shared = shared_memory.analyze_shared_memory(context.collector, context.buffer_facts, context.pass_configs)
        memory["assumptions"] += shared.pop("assumptions")
        memory.update(shared)
        waves = occupancy.analyze_waves(context.collector, memory, pressure, context.device_limits)
        tile_cost = combine_tile_cost(memory, waves)
        modules.update(memory_traffic=memory, waves=waves)
        trace.record("memory", lambda: memory)
        trace.record("waves", lambda: waves)

        pipeline_result = pipeline.analyze_pipeline(
            context.collector,
            memory,
            pressure,
            context.config.performance_model,
            context.pass_configs,
            loop=specialization.loop,
            family_name=specialization.name,
            phase_labels=phase_labels,
        )
        modules["pipeline_overlap"] = pipeline_result
        trace.record("pipeline", lambda: pipeline_result)
        ranking = apply_ranking_metric(tile_cost, waves, pipeline_result, context.config, specialization, pressure["register_demand"])
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


def analyze_kernel(func, config, target, device_limits, pass_configs, trace_context):
    """Execute the resolved analysis, retaining the established trace checkpoints."""
    with AnalysisTrace(config.trace_path) as trace:
        trace.record(
            "inputs",
            lambda: {
                "trace_context": trace_context,
                "target": str(target),
                "settings": config.to_cache_key_dict(),
                "device_limits": device_limits,
                "pass_configs": pass_configs,
            },
        )
        trace.record("prim_func", lambda: func.script())
        col = _Collector(func)
        trace.record("col", lambda: collector_snapshot(col))
        tile_propagation = _propagate_tiles(col, _kernel_outputs(col))
        trace.record("tile_propagation", lambda: propagation_snapshot(tile_propagation))
        buffer_facts = collect_buffer_facts(col)
        pressure = register_pressure.analyze_register_pressure(col, buffer_facts)
        trace.record("pressure.accumulator", lambda: pressure)
        context = AnalysisContext(
            func=func,
            collector=col,
            buffer_facts=buffer_facts,
            tile_propagation=tile_propagation,
            config=config,
            target=target,
            device_limits=device_limits,
            pass_configs=pass_configs,
            trace=trace,
        )
        results = run_modules(context, pressure)
        trace.record("tile_cost", lambda: results["tile_cost"])
        return {
            **results,
            "tile_propagation": tile_propagation.to_dict(),
            "ir_context": {
                "launch_threads": {k: str(v) for k, v in col.threads.items()},
                "explicit_layouts": {str(k): str(v) for k, v in col.layouts.items()},
            },
        }
