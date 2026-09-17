"""Tile pipeline work and a bounded-buffer overlap model.

No iteration unrolling and no timing fitted to candidate benchmarks. Timing is
available only with an explicit effective per-SM cost profile. Per-buffer ready
and reuse events model producer overlap. Instruction scheduling and CUDA
dispatch remain estimates.
"""

from .compute import compute_participants, consumer_threads, operation_work


from tiletune_core.pipeline import estimate_pipeline_cycles as estimate_pipeline_cycles


from tiletune_core.pipeline import _estimate_ampere as _estimate_ampere


def _analyze_single_pipeline(col, memory, pressure, performance_model=None, pass_configs=None, *, loop, family_name, phase_labels):
    from .src.ir_utils import _int
    from .src.ir_utils import loop_visits
    from .src.ir_utils import in_loop

    unknown = list(col.unknown)
    diagnostics = [dict(code="unresolved_ir", reason=reason) for reason in unknown]

    def add_unknown(code, reason, operation=None):
        unknown.append(reason)
        entry = dict(code=code, reason=reason)
        if operation is not None:
            entry["operation"] = operation
        diagnostics.append(entry)

    ampere = getattr(col, "ampere_plan", None)
    model = pressure.get("target_model")
    if model and model["kind"] is not None and not model["block_execution"]:
        add_unknown("unsupported_backend", "target requires its own core/storage scheduling model")
    if (
        model
        and model["kind"] not in (None, "cuda")
        and performance_model
        and (performance_model.get("profile_backend") != model["kind"] or performance_model.get("profile_target") != model["arch"])
    ):
        add_unknown("profile_mismatch", "non-CUDA timing requires an explicit matching profile_backend and profile_target")
    if ampere is None and any(
        op.kind == "elementwise" and in_loop(op, loop) and any(r.buffer.scope() == "global" for r in op.reads) for op in col.operations
    ):
        add_unknown(
            "unsupported_scalar_schedule", "direct scalar global accesses inside a recurrence require a per-iteration access schedule"
        )
    if performance_model and performance_model.get("profile_target", pressure.get("target_arch")) != pressure.get("target_arch"):
        add_unknown("profile_mismatch", "device profile target does not match the analyzed kernel")
    phases = [
        {
            "operation": op.index,
            "phase": phase_labels[op.index],
            "kind": op.kind,
            "dependencies": op.dependencies,
            "inside_loop": in_loop(op, loop),
            "work": operation_work(op, col),
            "compute_participants": compute_participants(op, pressure, pass_configs),
            "consumer_threads": consumer_threads(op),
        }
        for op in col.operations
    ]
    from .compute import reduction_work
    from .schedule import collect_cta_work
    from .schedule import collect_producer_buffers

    participants = {p["operation"]: p["compute_participants"] for p in phases}
    layout_cache = {}
    for op, phase in zip(col.operations, phases):
        for work, amount in phase["work"].items():
            if (
                work.startswith("convert_float32_to_float8")
                and amount
                and performance_model
                and not performance_model.get(work + "_per_cycle")
            ):
                add_unknown("missing_profile", f"operation {op.index} requires profile rate {work}_per_cycle", op.index)
        if op.index in getattr(col, "scalar_work_unknown", {}):
            add_unknown("unresolved_ownership", f"operation {op.index} scalar ownership: {col.scalar_work_unknown[op.index]}", op.index)
        if op.kind == "elementwise" and getattr(col, "inferred_layouts", {}).get(op.metadata.buffer.data) is not None:
            phase["scalar_work_basis"] = (
                "compiler fragment ownership and pure-expression reuse within each thread; address instructions excluded"
            )
        if ampere is not None:
            from .ampere import external_work

            phase["external_work"] = external_work(op)
            if any(value is None for value in phase["external_work"].values()):
                add_unknown("unresolved_external_access", f"operation {op.index} has unresolved external access bytes", op.index)
        phase["reduction"] = reduction_work(op, col, pressure, participants, layout_cache)
        reduction = phase["reduction"]
        if reduction and reduction["precision"] == "unknown":
            add_unknown(
                "unsupported_collective", f"operation {op.index} reduction: {reduction.get('reason', 'unresolved mapping')}", op.index
            )
        elif reduction and performance_model:
            from tiletune_core.profile_schema import reduction_rate_field

            if reduction_rate_field(reduction, performance_model, "local") is None:
                add_unknown("profile_mismatch", f"operation {op.index} reduction dtype does not match the profile", op.index)
            for primitive in ("local", "shuffle"):
                rate = reduction_rate_field(reduction, performance_model, primitive)
                if reduction[f"{primitive}_pairs"] and not performance_model.get(rate):
                    add_unknown("missing_profile", f"operation {op.index} requires profile rate {rate}", op.index)
    distribution = collect_cta_work(col, loop)
    try:
        producer_buffers, producer_unknown = collect_producer_buffers(col, loop), []
    except Exception as error:
        producer_buffers, producer_unknown = [], [str(error)]
    if performance_model and performance_model.get("gemm_signature"):
        signature = performance_model["gemm_signature"]
        if any(
            p["work"]["gemm_flops"] and any((p["compute_participants"] or {}).get(k) != v for k, v in signature.items()) for p in phases
        ):
            add_unknown("profile_mismatch", "device profile GEMM instruction/dtype signature does not match the kernel")
    iterations = {"min": None, "max": None, "precision": "unknown"}
    stages = None
    if loop is None and not col.pipeline_loops and not col.serial_loops:
        # A single-pass tile has only outside-loop work. Zero recurrence steps
        # let the same timing equation count that work exactly once.
        stages = 0
        iterations = {"min": 0, "max": 0, "precision": "exact"}
    elif loop is None:
        add_unknown("multiple_regions", "no recognized single tile pipeline loop")
    else:
        stages = _int(loop.annotations.get("num_stages", 0))
        representative = next(op for op in col.operations if in_loop(op, loop))
        domains = tuple(entry for entry in representative.loops if entry[2] == "4" or entry[0].same_as(loop.loop_var))
        iterations = loop_visits(domains)
        if iterations["max"] is None or stages is None:
            add_unknown("unresolved_loop", "unresolved loop count or buffer depth")
    if any(
        kind not in ("4", "1") and (loop is None or not var.same_as(loop.loop_var)) for op in col.operations for var, _, kind in op.loops
    ):
        add_unknown("nested_regions", "nested serial loop scheduling is not modeled")
    name = str(loop.loop_var) if loop is not None else None
    repeating = [tile for tile in memory["input_tiles"] if name in tile.get("loop_variables", [])]
    once = [tile for tile in memory["input_tiles"] if name not in tile.get("loop_variables", [])]
    if any(tile["tile_bytes"] is None or tile["bytes_per_block"] is None for tile in memory["input_tiles"] + memory["output_tiles"]):
        add_unknown("unresolved_memory_bounds", "unresolved memory tile size")
    if any(value is None for phase in phases for value in phase["work"].values()):
        add_unknown("unresolved_work", "unresolved operation work")
    ws = pressure.get("warp_specialization", {})
    eligible = ws.get("status") == "predicted"
    if ampere is not None and stages:
        eligible = ampere["status"] == "predicted"
        for reason in ampere["unknown"]:
            add_unknown("unsupported_scheduling", reason)
        if performance_model:
            for field in ("async_copy_issue_bytes_per_cycle", "async_copy_latency_cycles"):
                if field not in performance_model:
                    add_unknown("missing_profile", f"Ampere asynchronous pipeline requires profile field {field}")
    if stages and not eligible:
        add_unknown("unresolved_pipeline_policy", "positive-stage pipeline scheduling policy is unresolved or unsupported")
    if any(op.predicates for op in col.operations):
        add_unknown("branch_regions", "branch-dependent operation schedule")

    def known_sum(values):
        return sum(values) if all(value is not None for value in values) else None

    result = {
        "specialization": family_name,
        "ampere_schedule": ampere,
        "phases": phases,
        "cta_work": distribution,
        "producer_buffers": producer_buffers,
        "producer_schedule_unknown": producer_unknown,
        "iterations": iterations,
        "num_stages": stages,
        "effective_buffer_depth": max(1, stages) if stages is not None else None,
        "overlap_eligible": eligible,
        "producer_copies_per_iteration": len(repeating),
        "input_bytes_per_iteration": known_sum([t["tile_bytes"] for t in repeating]),
        "outside_loop_input_copies": len(once),
        "outside_loop_bytes": known_sum([t["bytes_per_block"] for t in once] + [memory["output_bytes_per_block"]]),
        "loop_carried_buffers": (pressure.get("tile_liveness") or {}).get("loop_carried_buffers", []),
        "performance_model": performance_model,
        "unknown": sorted(set(unknown)),
        "diagnostics": diagnostics,
        "precision": "unknown" if unknown else "conservative" if iterations["precision"] != "exact" else "estimate",
        "assumptions": [
            "each producer tile is waited on at first use and released after its last consumer",
            "max-plus recurrence models separate buffer rings and shared byte-service capacity",
            "a finite buffer ring limits overlap; loop-carried consumer state serializes consumer iterations",
            "no loop unrolling; startup and repeated steady-state intervals are modeled analytically",
            "CTA timing reports maximum work; grid ranking uses the separately collected per-CTA work distribution",
            "explicit profile rates are effective per-SM rates for the target and operation dtypes; no compiler counters or benchmark latencies",
            "optional WGMMA per-warpgroup rate limits a CTA's compute service independently of the aggregate per-SM rate",
            "optional single-CTA primitive rates limit scalar/exp/reduction service at the original consumer thread count",
            "consumer probes use eight independent chains; actual instruction dependencies and phase occupancy can differ",
        ],
    }
    result["timing"] = estimate_pipeline_cycles(result)
    result["timing_status"] = "estimate" if result["timing"] is not None else "unknown"
    return result


def analyze_pipeline(col, memory, pressure, performance_model=None, pass_configs=None, *, loop, family_name, phase_labels):
    result = _analyze_single_pipeline(
        col, memory, pressure, performance_model, pass_configs, loop=loop, family_name=family_name, phase_labels=phase_labels
    )
    needs_regions = len(col.serial_loops + col.pipeline_loops) > 1 or any(op.predicates for op in col.operations)
    needs_regions |= any(item["code"] == "unresolved_external_access" for item in result["diagnostics"])
    if not needs_regions:
        return result
    from .region_schedule import UnresolvedRegion, build_region_schedule, region_totals
    from .schedule import collect_producer_buffers

    try:
        schedule = build_region_schedule(col, result["phases"], pressure)
        producers = [copy for region_loop in col.pipeline_loops + col.serial_loops for copy in collect_producer_buffers(col, region_loop)]
    except UnresolvedRegion as error:
        result["diagnostics"].insert(0, dict(code=error.code, reason=str(error)))
        return result
    except ValueError as error:
        result["diagnostics"].insert(0, dict(code="unsupported_scheduling", reason=str(error)))
        return result
    superseded = {
        "multiple_regions",
        "nested_regions",
        "branch_regions",
        "unresolved_memory_bounds",
        "unresolved_loop",
        "unresolved_pipeline_policy",
        "unresolved_external_access",
    }
    result["diagnostics"] = [item for item in result["diagnostics"] if item["code"] not in superseded]
    result["unknown"] = sorted({item["reason"] for item in result["diagnostics"]})
    result.update(region_schedule=schedule, region_producers=producers, cta_work=schedule["cta_work"], precision="estimate")
    totals = [region_totals(body) for body in schedule["variants"]]
    result["region_work"] = totals
    result["region_memory"] = dict(
        input_bytes_per_block=max(t["read_bytes"] for t in totals),
        output_bytes_per_block=max(t["write_bytes"] for t in totals),
        traffic_bytes_per_block=max(t["read_bytes"] + t["write_bytes"] for t in totals),
        cta_work_uniform=len(totals) == 1,
        unknown=list(col.unknown),
        precision="exact",
    )
    result["timing"] = estimate_pipeline_cycles(result)
    result["timing_status"] = "estimate" if result["timing"] is not None else "unknown"
    return result
