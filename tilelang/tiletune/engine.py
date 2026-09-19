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
    memory_mode = context.config.ranking_metric == "memory"
    if memory_mode:
        from .families.base import KernelSpecialization

        specialization = KernelSpecialization()
    else:
        specialization = select_specialization(context.collector, context.config.specialization)
    trace.record("specialization", lambda: specialization.to_dict())

    phase_labels = {op.index: specialization.phase(op) for op in context.collector.operations}
    pressure["tile_liveness"] = analyze_live_tiles(context.collector, context.buffer_facts, loop=specialization.loop)
    if hasattr(context.collector, "ampere_plan"):
        from .ampere import operand_registers

        pressure["ampere_mma_operand_registers"] = operand_registers(context.collector, context.target)
    for phase in pressure["tile_liveness"]["phases"]:
        phase["phase"] = phase_labels[phase["operation"]]
    trace.record("pressure.tile_liveness", lambda: pressure["tile_liveness"])
    pressure.update(resolve_register_budget(context.config, context.target))
    ws = (
        {"status": "not_modeled", "reason": "memory ordering does not require a compiler scheduling policy"}
        if memory_mode
        else predict_warp_specialization(
            context.func, context.collector, pressure, context.pass_configs, policy=specialization.warp_specialization_policy()
        )
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
    if context.config.ranking and memory_mode:
        from .memory import analyze_memory_accesses
        from tiletune_core.memory import score_memory

        memory = analyze_memory_accesses(context.collector, context.buffer_facts)
        shared = shared_memory.analyze_shared_memory(context.collector, context.buffer_facts, context.pass_configs)
        ranking = score_memory(memory["accesses"], memory["grid_blocks"], (context.device_limits or {}).get("sm_count"))
        # An opaque operation may hide memory effects not present in the ledger.
        if memory["unknown"]:
            ranking.update(score=None, tie_break_score=None, precision="unknown")
            ranking["unknown"].extend(memory["unknown"])
        tile_cost = dict(
            **memory,
            **shared,
            score=ranking["score"],
            tie_break_score=ranking["tie_break_score"],
            score_formula=ranking["formula"],
            ranking_metric="memory",
            precision=ranking["precision"],
        )
        tile_cost["unknown"] = ranking["unknown"]
        modules.update(memory_traffic=memory, shared_memory=shared, ranking=ranking)
        trace.record("memory", lambda: memory)
        trace.record("shared_memory", lambda: shared)
        trace.record("ranking", lambda: ranking)
        if context.config.facts_path:
            import json
            from pathlib import Path

            facts = dict(version=1, backend="memory.v1", **memory, sm_count=(context.device_limits or {}).get("sm_count"))
            path = Path(context.config.facts_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(facts, indent=2, allow_nan=False) + "\n")
    elif context.config.ranking:
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
        if "region_memory" in pipeline_result:
            memory.update(pipeline_result["region_memory"])
        workspace = max(((p.get("reduction") or {}).get("workspace_bytes", 0) for p in pipeline_result["phases"]), default=0)
        memory["collective_workspace_bytes_estimate"] = workspace
        if memory["shared_memory_bytes_estimate"] is not None:
            memory["shared_memory_bytes_estimate"] += workspace
        if workspace:
            memory["assumptions"].append(
                "serial scalar collectives reuse one thread-sized workspace; workspace remains separate from the tile arena"
            )
        waves = occupancy.analyze_waves(context.collector, memory, pressure, context.device_limits)
        tile_cost = combine_tile_cost(memory, waves)
        modules.update(memory_traffic=memory, waves=waves)
        trace.record("memory", lambda: memory)
        trace.record("waves", lambda: waves)

        modules["pipeline_overlap"] = pipeline_result
        trace.record("pipeline", lambda: pipeline_result)
        target_kind = (pressure.get("target_model") or {}).get("kind")
        if target_kind == "cuda":
            from .facts import cuda_facts
            from tiletune_core.cuda import evaluate_cuda_facts

            facts = cuda_facts(memory, waves, pipeline_result, context.config, specialization, pressure)
            evaluation = evaluate_cuda_facts(facts)
            ranking = evaluation.details["ranking"]
            trace.record("resolved_facts", facts.to_dict)
            if context.config.facts_path:
                import json
                from pathlib import Path

                path = Path(context.config.facts_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(facts.to_dict(), indent=2, allow_nan=False) + "\n")
        else:
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
    from .diagnostics import analysis_diagnostics

    return {
        "specialization": specialization.to_dict(),
        "modules": modules,
        "pressure": pressure,
        "tile_cost": tile_cost,
        "diagnostics": analysis_diagnostics(modules),
    }


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
        col = _Collector(func, input_values=config.input_values)
        trace.record("col", lambda: collector_snapshot(col))
        from .ampere import is_ampere, prepare_analysis

        if config.ranking and config.ranking_metric == "pipeline_time" and is_ampere(target):
            prepare_analysis(func, col, target, pass_configs)
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
