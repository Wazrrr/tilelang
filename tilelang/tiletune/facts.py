"""TIR extraction adapter for the compiler-independent fact contract."""

from tiletune_core import KernelFacts as KernelFacts


def _resolved(value):
    # Internal legacy reports use integer operation keys. JSON serializes them
    # as strings. Do so explicitly; reject every nonprimitive leaf in the core.
    if isinstance(value, dict):
        return {str(k): _resolved(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_resolved(v) for v in value]
    if isinstance(value, str):
        return str(value)
    return value


def cuda_facts(memory, waves, pipeline, config, specialization, pressure):
    """Only resolved numeric/JSON analysis records cross this boundary."""
    return KernelFacts(
        backend="cuda.v21",
        target={"kind": "cuda", "arch": pressure.get("target_arch")},
        operations=_resolved(pipeline["phases"]),
        regions=_resolved([dict(body=body) for body in pipeline.get("region_schedule", {}).get("variants", [])]),
        ownership=_resolved([p.get("compute_participants") or {} for p in pipeline["phases"]]),
        storage=_resolved(memory.get("shared_allocations", [])),
        dependencies=_resolved([dict(operation=p["operation"], predecessors=p["dependencies"]) for p in pipeline["phases"]]),
        launch=dict(grid_blocks=waves["grid_blocks"], launch_threads=waves["launch_threads"]),
        payload=_resolved(
            dict(
                memory=memory,
                waves=waves,
                pipeline=pipeline,
                config=dict(ranking_metric=config.ranking_metric, performance_model=config.performance_model),
                specialization_matched=specialization.matched,
                register_demand=pressure["register_demand"],
            )
        ),
    )
