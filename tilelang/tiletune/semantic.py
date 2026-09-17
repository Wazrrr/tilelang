"""Recover exact workload facts hidden behind runtime dispatch metadata.

Most kernels are modeled directly from TIR. A grouped GEMM is exceptional: the
group-offset tensors select addresses at runtime, but their shapes and the
generated padded CTA domain remain visible in TIR. This module reconstructs the
schedule from that executable IR without annotations or candidate measurements.
"""

import math


def _grouped_gemm_metadata(func, collector):
    from .src.ir_utils import _int, dense_gemms, main_loops

    names = {buffer.name for buffer in func.buffer_map.values()}
    if not {"batch_sizes", "batch_offsets", "batch_padded_offsets"} <= names:
        return None
    gemms = dense_gemms(collector)
    loops = main_loops(collector, gemms)
    if len(gemms) != 1 or len(loops) != 1:
        return None
    gemm = gemms[0]
    buffers = {buffer.name: buffer for buffer in func.buffer_map.values()}
    a = next(buffer for name, buffer in buffers.items() if name.startswith("A"))
    c = next(buffer for name, buffer in buffers.items() if name.startswith("C"))
    sizes = next(buffer for name, buffer in buffers.items() if name.startswith("batch_sizes"))
    return dict(
        kind="grouped_gemm",
        group_count=_int(sizes.shape[0]),
        m_blocks=_int(collector.threads["blockIdx.x"]),
        n=_int(c.shape[1]),
        k=_int(a.shape[1]),
        dtype=str(a.dtype),
        block_m=_int(gemm.metadata.cRegion.region[0].extent),
        block_n=_int(gemm.metadata.cRegion.region[1].extent),
        block_k=_int(gemm.metadata.aRegion.region[1].extent),
        num_stages=_int(loops[0].annotations.get("num_stages", 0)),
        threads=_int(collector.threads["threadIdx.x"]),
    )


def _resident_blocks(limits, shared_bytes, register_words, threads):
    required = {
        "sm_count",
        "shared_memory_per_sm",
        "registers_per_sm",
        "max_threads_per_sm",
        "max_blocks_per_sm",
        "warp_size",
    }
    if not required <= limits.keys():
        return None, {}, ["incomplete occupancy inputs"]
    warp = limits["warp_size"]
    rounded_threads = math.ceil(threads / warp) * warp
    bounds = {
        "threads": limits["max_threads_per_sm"] // rounded_threads,
        "shared_memory": limits["shared_memory_per_sm"] // max(1, shared_bytes),
        "registers": limits["registers_per_sm"] // max(1, register_words),
        "blocks": limits["max_blocks_per_sm"],
    }
    unknown = []
    if shared_bytes > limits.get("shared_memory_per_block", shared_bytes) or threads > limits.get("max_threads_per_block", threads):
        unknown.append("estimated block resources exceed device limits")
    resident = min(bounds.values())
    if resident <= 0:
        unknown.append("unresolved wave count")
        resident = None
    return resident, bounds, unknown


def _grouped_gemm(metadata, config, performance_model, device_limits, pressure):
    bm, bn, bk = (metadata[key] for key in ("block_m", "block_n", "block_k"))
    n, k, original_threads = metadata["n"], metadata["k"], metadata["threads"]
    stages = metadata["num_stages"]
    group_count = metadata["group_count"]
    element_bytes = 2 if metadata["dtype"] in ("float16", "bfloat16") else 4
    n_tiles = math.ceil(n / bn)
    grid = metadata["m_blocks"] * n_tiles
    iterations = math.ceil(k / bk)
    depth = max(1, stages)
    input_bytes = (bm * bk + bn * bk) * element_bytes
    output_bytes = bm * bn * element_bytes
    metadata_bytes = (group_count + 3) * 4
    shared_bytes = input_bytes * depth
    threads = original_threads
    logical_register_words = bm * bn
    register_words = logical_register_words
    register_basis = "exact FP32 accumulator tile plus native MMA operand fragments; compiler scratch remains unknown"
    operands = pressure.get("mma_operand_registers", {})
    operand_unknown = []
    operand_estimates = []
    for phase in (pressure.get("tile_liveness") or {}).get("phases", []):
        temporary = operands.get(phase["operation"], {})
        if temporary.get("unknown"):
            operand_unknown.append("unresolved MMA operand register storage")
        live = phase.get("packed_registers_per_block_estimate")
        if live is not None:
            operand_estimates.append(live + temporary.get("registers_per_block", 0))
    register_words = max([register_words, *operand_estimates])
    ws = pressure.get("warp_specialization", {})
    if ws.get("status") == "predicted":
        threads = ws["launch_threads"]
        register_words = ws["register_reservation_per_block"]
        register_basis = "producer/consumer policy reservation; independent of logical tile demand"
    elif ws.get("status") == "unknown":
        operand_unknown.append("unresolved automatic warp-specialization policy")
    limits = dict(device_limits or {})
    resident, bounds, unknown = _resident_blocks(limits, shared_bytes, register_words, threads)
    unknown.extend(operand_unknown)
    waves_count = math.ceil(grid / (resident * limits["sm_count"])) if resident else None
    waves = {
        "grid_blocks": grid,
        "launch_threads": threads,
        "original_launch_threads": original_threads,
        "registers_per_block_estimate": register_words,
        "logical_tile_registers_per_block_estimate": logical_register_words,
        "register_estimate_basis": register_basis,
        "device_limits": limits,
        "resident_blocks_limits": bounds,
        "resident_blocks_per_sm_estimate": resident,
        "num_waves_estimate": waves_count,
        "unknown": unknown,
        "precision": "unknown" if unknown else "estimate",
        "assumptions": [
            "runtime group dispatch changes addresses but not the full padded CTA tensor-core work",
            "register occupancy includes native MMA operand fragments or a recognized producer/consumer reservation",
        ],
        "implementation": "grouped_gemm_metadata",
    }
    memory = {
        "input_tiles": [],
        "output_tiles": [],
        "input_bytes_per_block": iterations * input_bytes + metadata_bytes,
        "output_bytes_per_block": output_bytes,
        "traffic_bytes_per_block": iterations * input_bytes + output_bytes + metadata_bytes,
        "shared_allocations": [],
        "shared_memory_bytes_estimate": shared_bytes,
        "shared_memory_allocated_sum_bytes": shared_bytes,
        "unknown": [],
        "precision": "exact",
        "assumptions": [
            "metadata reads include every padded-offset dispatch test plus selected size/offset/padded-offset values",
            "tail CTAs execute a full accumulator tile; masked global edges can only reduce transferred bytes",
        ],
        "implementation": "grouped_gemm_metadata",
    }
    pipeline = {
        "specialization": "grouped_gemm",
        "iterations": {"min": iterations, "max": iterations, "precision": "exact"},
        "num_stages": stages,
        "effective_buffer_depth": depth,
        "input_bytes_per_iteration": input_bytes,
        "outside_loop_bytes": output_bytes + metadata_bytes,
        "cta_work": {"kind": "uniform", "visits": iterations, "blocks": grid},
        "performance_model": performance_model,
        "unknown": [],
        "diagnostics": [],
        "precision": "estimate",
        "assumptions": [
            "the builder's immutable group sizes recover the runtime dispatch domain exactly",
            "positive-stage execution overlaps copy service and GEMM service after the first iteration",
        ],
        "implementation": "grouped_gemm_metadata",
    }
    ranking = {
        "metric": config.ranking_metric,
        "score": None,
        "precision": "unknown",
        "formula": None,
        "unknown": [],
    }
    if config.ranking_metric == "traffic_waves":
        ranking.update(
            score=(memory["traffic_bytes_per_block"] + 1) * waves_count if waves_count and not unknown else None,
            precision="estimate" if waves_count and not unknown else "unknown",
            formula="(traffic_bytes_per_block + 1) * num_waves_estimate",
            units="byte-waves",
        )
    elif not performance_model:
        ranking["unknown"].append("pipeline_time requires an explicit device profile")
    elif unknown:
        ranking["unknown"].extend(unknown)
    else:
        active = min(resident, math.ceil(grid / limits["sm_count"]))
        global_rate = performance_model["global_bytes_per_cycle"] / active
        gemm_rate = performance_model["gemm_flops_per_cycle"] / active
        shared_rate = performance_model["shared_bytes_per_cycle"] / active
        copy_service = input_bytes / global_rate
        copy_ready = copy_service
        gemm_flops = 2 * bm * bn * bk
        gemm_service = max(gemm_flops / gemm_rate, input_bytes / shared_rate)
        group_rate = performance_model.get("wgmma_flops_per_cycle_per_warpgroup")
        if group_rate:
            gemm_service = max(gemm_service, gemm_flops / (max(1, threads // 128) * group_rate))
        barrier = performance_model["barrier_cycles"]
        consumer = gemm_service + 2 * barrier
        if stages:
            from tiletune_core.schedule import buffer_transition, repeat_transition

            copies = [
                {"operation": 0, "bytes": bm * bk * element_bytes, "first_consumer": 2, "last_consumer": 2},
                {"operation": 1, "bytes": bn * bk * element_bytes, "first_consumer": 2, "last_consumer": 2},
            ]
            transition = buffer_transition(
                copies,
                [(2, gemm_service)],
                stages,
                global_rate,
                performance_model["copy_latency_cycles"],
                barrier,
            )
            loop_cycles = repeat_transition(transition, iterations)[0]
            first = repeat_transition(transition, 1)[0]
            previous = repeat_transition(transition, max(0, iterations - 1))[0]
            step = loop_cycles - previous if iterations else 0
            copy_ready += performance_model["copy_latency_cycles"]
            schedule = "grouped GEMM per-buffer max-plus recurrence"
        else:
            step = copy_ready + consumer
            first = step
            loop_cycles = iterations * step
            schedule = "grouped GEMM serial copy/consumer loop"
        outside = (output_bytes + metadata_bytes) / global_rate
        cta_cycles = loop_cycles + outside
        score = cta_cycles * waves_count * performance_model.get("latency_scale", 1)
        clock = performance_model.get("reference_clock_mhz")
        timing = {
            "cycles": cta_cycles,
            "consumer_cycles_per_iteration": consumer,
            "copy_service_cycles_per_iteration": copy_service,
            "input_ready_latency_cycles": copy_ready,
            "steady_state_interval_cycles": step,
            "fill_and_first_consumer_cycles": first,
            "iterations": iterations,
            "schedule_model": schedule,
            "outside_loop_cycles": outside,
            "concurrent_ctas": active,
        }
        pipeline.update(timing=timing, timing_status="estimate")
        ranking.update(
            score=score,
            precision="estimate",
            formula="num_waves_estimate * grouped_gemm_cta_pipeline_cycles",
            units="cycles",
            wave_timing=timing,
            estimated_latency_ms=score / (1000 * clock) if clock else None,
        )
    if ranking["score"] is None and not ranking["unknown"]:
        ranking["unknown"].append("grouped GEMM semantic model lacks complete profile or occupancy inputs")
    tile_cost = {
        **memory,
        **waves,
        "traffic_bytes_grid_estimate": memory["traffic_bytes_per_block"] * grid,
        "score": ranking["score"],
        "score_formula": ranking["formula"],
        "ranking_metric": ranking["metric"],
        "precision": ranking["precision"],
        "unknown": ranking["unknown"],
        "assumptions": memory["assumptions"] + waves["assumptions"],
    }
    return dict(memory=memory, waves=waves, pipeline=pipeline, ranking=ranking, tile_cost=tile_cost)


def analyze_semantic_workload(func, collector, config, performance_model, device_limits, pressure):
    metadata = _grouped_gemm_metadata(func, collector)
    if metadata is None:
        return None
    if metadata["kind"] == "grouped_gemm":
        return _grouped_gemm(metadata, config, performance_model, device_limits, pressure)
    return None
