"""Capture program facts, estimate storage, and run the requested ranking model.

Memory ranking uses operator semantics and backend resource inputs only. The
legacy timing path retains its family policies. Unexpected stage errors propagate.
"""

from dataclasses import dataclass
from math import prod

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


# Datasheet roofline references for diagnostics, not ranking gates.
BOUND_RIDGE_FLOPS_PER_BYTE = {
    "ampere": 200.0,
    "sm_100a": 281.25,
    "sm_103": 281.25,
    "sm_103a": 281.25,
}


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


def _strict_register_analysis(config):
    """Return whether a proven accumulator bound may reject before lowering."""
    return config.register_cap is not None or config.max_spill_bytes == 0 or config.max_local_bytes == 0


def _launch_only_liveness(col):
    """Retain the hard launch-size check without computing register intervals."""
    from .src.ir_utils import _int

    dimensions = [_int(value) for name, value in col.threads.items() if name.startswith("threadIdx.")]
    threads = prod(dimensions) if dimensions and all(value is not None and value > 0 for value in dimensions) else None
    return {
        "precision": "disabled",
        "phases": [],
        "peak_registers_per_block_estimate": None,
        "computing_threads_estimate": threads,
        "loop_carried_buffers": [],
        "assumptions": [
            "register tile liveness is disabled in lean memory mode",
            "launch threads remain available for the hard device block limit",
            "set memory_diagnostics=True to report register tile intervals",
        ],
    }


def _disabled_register_pressure():
    return {
        "logical_storage": [],
        "modeled_lower_bound": None,
        "modeled_accumulator_registers_per_block": None,
        "total_register_upper_bound": None,
        "evidence": [],
        "assumptions": [
            "register storage analysis is disabled because no strict pre-lowering register policy was requested",
            "compiler allocation and physical limits remain subject to post-compile validation",
            "set memory_diagnostics=True to report logical register storage",
        ],
    }


def _disabled_shared_memory():
    return {
        "precision": "disabled",
        "shared_allocations": [],
        "shared_memory_bytes_estimate": None,
        "shared_memory_allocated_sum_bytes": None,
        "shared_storage_plan": {
            "precision": "disabled",
            "allocated_sum_bytes": None,
            "arena_bytes_estimate": None,
            "intervals": [],
            "reuse_bytes_estimate": None,
            "method": "disabled in lean memory mode",
            "unknown": [],
        },
        "assumptions": ["set memory_diagnostics=True to report shared-memory allocation lifetimes"],
    }


def _disabled_tile_propagation():
    return {
        "precision": "disabled",
        "operations": [],
        "per_iteration_inputs": [],
        "full_loop_inputs": [],
        "unknown": [],
        "reason": "not required by lean memory ranking; set memory_diagnostics=True to report it",
    }


def run_modules(context, pressure):
    """Run the common stages and make one final register decision."""
    from .ranking import apply_ranking_metric

    trace = context.trace
    memory_mode = context.config.ranking_metric in ("memory", "bound_aware", "rank_product", "work_max", "work_rank_product")
    detailed_memory = memory_mode and context.config.memory_diagnostics
    if memory_mode:
        specialization = None
        specialization_info = {
            "name": "generic",
            "matched": True,
            "loop": None,
            "roles": {},
            "evidence": ["IR operator facts and backend inputs; no kernel-family policy"],
        }
        phase_labels = {op.index: op.kind for op in context.collector.operations}
        loop, spill_allowance = None, 0
    else:
        specialization = select_specialization(context.collector, context.config.specialization)
        specialization_info = specialization.to_dict()
        phase_labels = {op.index: specialization.phase(op) for op in context.collector.operations}
        loop = specialization.loop
        spill_allowance = specialization.register_spill_allowance(context.config)
    trace.record("specialization", lambda: specialization_info)

    pressure["tile_liveness"] = (
        analyze_live_tiles(context.collector, context.buffer_facts, loop=loop)
        if not memory_mode or detailed_memory
        else _launch_only_liveness(context.collector)
    )
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
    from .targets import resolve_target

    target_model = resolve_target(context.target)
    if (
        not memory_mode
        and ws.get("status") != "predicted"
        and target_model.kind == "cuda"
        and target_model.architecture in ("ampere", "blackwell")
    ):
        from .ampere import operand_registers

        pressure["mma_operand_registers"] = operand_registers(context.collector, context.target)
        if target_model.architecture == "ampere":
            pressure["ampere_mma_operand_registers"] = pressure["mma_operand_registers"]
    pressure.update(analyze_register_policy(pressure, context.config, context.device_limits, spill_allowance=spill_allowance))
    trace.record(
        "pressure.register_policy", lambda: {key: pressure[key] for key in ("physical_register_allocation", "register_demand", "decision")}
    )
    modules = {"register_pressure": pressure, "warp_specialization": ws}

    tile_cost = {"score": None, "precision": "disabled"}
    modules.update({name: {"precision": "disabled"} for name in ("memory_traffic", "waves", "pipeline_overlap", "ranking")})
    if context.config.ranking and memory_mode:
        from .memory import analyze_compute_intensity, analyze_memory_accesses
        from tiletune_core.memory import classify_bound, score_memory, score_rank_product

        memory = analyze_memory_accesses(
            context.collector, context.buffer_facts, include_dependencies=detailed_memory
        )
        shared = (
            shared_memory.analyze_shared_memory(context.collector, context.buffer_facts, context.pass_configs)
            if detailed_memory
            else _disabled_shared_memory()
        )
        if context.config.ranking_metric == "bound_aware":
            bound = analyze_compute_intensity(context.collector, context.buffer_facts, memory["grid_blocks"])
            target_model = pressure.get("target_model") or {}
            ridge = BOUND_RIDGE_FLOPS_PER_BYTE.get(
                target_model.get("arch"), BOUND_RIDGE_FLOPS_PER_BYTE.get(target_model.get("architecture"))
            )
            bound["bound"] = (
                classify_bound(bound["compute_work"], bound["unique_global_bytes"], ridge)
                if ridge is not None
                else None
            )
            bound["ridge_flops_per_byte"] = ridge
            bound["occupancy_gate_enabled"] = False
            bound["occupancy_penalty"] = 1
            modules["bound"] = bound
            trace.record("bound", lambda: bound)
        if context.config.ranking_metric in ("work_max", "work_rank_product"):
            from .work_max import analyze_compute_work
            from tiletune_core.work_max import score_work_max, score_work_rank_product

            compute = analyze_compute_work(context.collector, pressure, context.pass_configs)
            modules["compute_work"] = compute
            trace.record("compute_work", lambda: compute)
            score_work = score_work_rank_product if context.config.ranking_metric == "work_rank_product" else score_work_max
            ranking = score_work(
                memory["accesses"], compute, memory["grid_blocks"],
                (context.device_limits or {}).get("sm_count"), context.config.performance_model,
                memory["pipeline_depth"],
            )
        elif context.config.ranking_metric == "rank_product":
            ranking = score_rank_product(
                memory["accesses"],
                memory["grid_blocks"],
                (context.device_limits or {}).get("sm_count"),
                memory["pipeline_depth"],
            )
        else:
            ranking = score_memory(
                memory["accesses"],
                memory["grid_blocks"],
                (context.device_limits or {}).get("sm_count"),
                memory["pipeline_depth"],
                launch_underfill=context.config.ranking_metric == "bound_aware",
            )
        ranking["metric"] = context.config.ranking_metric
        if memory["unknown"]:
            ranking.update(score=None, tie_break_score=None, precision="unknown")
            ranking["unknown"].extend(memory["unknown"])
            if "component_scores" in ranking:
                ranking["component_scores"] = dict.fromkeys(ranking["component_scores"])
        tile_cost = {
            **memory,
            **shared,
            "score": ranking["score"],
            "tie_break_score": ranking["tie_break_score"],
            "logical_byte_waves": ranking["logical_byte_waves"],
            "logical_memory_access_waves": ranking["logical_memory_access_waves"],
            "score_formula": ranking["formula"],
            "ranking_metric": context.config.ranking_metric,
            "precision": ranking["precision"],
            "unknown": ranking["unknown"],
        }
        if "component_scores" in ranking:
            tile_cost.update(component_scores=ranking["component_scores"], score_scope=ranking["score_scope"])
        if context.config.ranking_metric in ("work_max", "work_rank_product"):
            tile_cost.update(service_cycles=ranking["service_cycles"], normalized_work=ranking["adjusted_logical_byte_waves"])
        modules.update(memory_traffic=memory, shared_memory=shared, ranking=ranking)
        trace.record("memory", lambda: memory)
        trace.record("shared_memory", lambda: shared)
        trace.record("ranking", lambda: ranking)
        if context.config.facts_path:
            import json
            from pathlib import Path

            backend = {
                "memory": "memory.v2",
                "bound_aware": "bound_aware.v1",
                "rank_product": "rank_product.v1",
                "work_max": "work_max.v1",
                "work_rank_product": "work_rank_product.v1",
            }[context.config.ranking_metric]
            facts = {"version": 1, "backend": backend, **memory, "sm_count": (context.device_limits or {}).get("sm_count")}
            if context.config.ranking_metric == "bound_aware":
                facts["bound"] = modules["bound"]
            if context.config.ranking_metric in ("work_max", "work_rank_product"):
                facts.update(compute=modules["compute_work"], performance_model=context.config.performance_model)
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
        from .semantic import analyze_semantic_workload

        semantic = analyze_semantic_workload(
            context.func,
            context.collector,
            context.config,
            context.config.performance_model,
            context.device_limits,
            pressure,
        )
        if semantic is not None:
            memory = semantic["memory"]
            waves = semantic["waves"]
            pipeline_result = semantic["pipeline"]
            ranking = semantic["ranking"]
            tile_cost = semantic["tile_cost"]
            modules.update(memory_traffic=memory, waves=waves, pipeline_overlap=pipeline_result)
            trace.record("semantic_workload", lambda: semantic)
        elif (pressure.get("target_model") or {}).get("kind") == "cuda":
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
        tile_cost.update(
            score=ranking["score"],
            score_formula=ranking["formula"],
            ranking_metric=ranking["metric"],
            score_relative_uncertainty=ranking.get("score_relative_uncertainty", 0),
        )
        if ranking["score"] is None:
            tile_cost["precision"] = "unknown"
            tile_cost["unknown"] = sorted(set(tile_cost["unknown"] + ranking["unknown"]))
    else:
        trace.record("ranking", lambda: {"precision": "disabled", "reason": "config.ranking is False"})

    for name, result in modules.items():
        result["implementation"] = specialization_info["name"] if name != "waves" else "generic"
    from .diagnostics import analysis_diagnostics

    return {
        "specialization": specialization_info,
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
        memory_mode = config.ranking_metric in ("memory", "bound_aware", "rank_product", "work_max", "work_rank_product")
        detailed_memory = memory_mode and config.memory_diagnostics
        strict_registers = memory_mode and _strict_register_analysis(config)
        col = _Collector(
            func,
            input_values=config.input_values,
            track_dependencies=not memory_mode or detailed_memory,
            memory_only=memory_mode and not detailed_memory,
        )
        if col.memory_only and col.input_values and col.unknown:
            # Eager simplification can prove an unreachable opaque access or a
            # metadata index safe. Retry it before declaring scored work unknown.
            col = _Collector(func, input_values=config.input_values, track_dependencies=False)
        trace.record("col", lambda: collector_snapshot(col))
        from .ampere import is_ampere, prepare_analysis

        if config.ranking and config.ranking_metric == "pipeline_time" and is_ampere(target):
            prepare_analysis(func, col, target, pass_configs)
        elif config.ranking and config.ranking_metric == "pipeline_time":
            from .targets import resolve_target

            if resolve_target(target).architecture == "blackwell" and any(op.kind == "reduce" for op in col.operations):
                from .ampere import prepare_ownership_analysis

                prepare_ownership_analysis(func, col, target, pass_configs)
        accumulator_check = strict_registers and any(
            hasattr(op.metadata, "cRegion") and not bool(getattr(op.metadata, "isTcgen05", False))
            for op in col.operations
        )
        tile_propagation = (
            _propagate_tiles(col, _kernel_outputs(col))
            if not memory_mode or detailed_memory or accumulator_check
            else None
        )
        trace.record(
            "tile_propagation",
            lambda: (
                propagation_snapshot(tile_propagation)
                if tile_propagation is not None
                else _disabled_tile_propagation()
            ),
        )
        buffer_facts = collect_buffer_facts(col)
        pressure = (
            register_pressure.analyze_register_pressure(col, buffer_facts)
            if not memory_mode or detailed_memory or strict_registers
            else _disabled_register_pressure()
        )
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
            "tile_propagation": (
                tile_propagation.to_dict()
                if tile_propagation is not None and (not memory_mode or detailed_memory)
                else _disabled_tile_propagation()
            ),
            "ir_context": {
                "metadata_resolution": "not_needed"
                if not col.input_values
                else "deferred"
                if col.memory_only
                else "eager",
                "launch_threads": {k: str(v) for k, v in col.threads.items()},
                "explicit_layouts": {str(k): str(v) for k, v in col.layouts.items()},
            },
        }
