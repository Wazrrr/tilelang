"""Simple Carver model for the native SM100 grouped-MXFP8 schedules."""

import math
import re


_CONFIG_KEYS = {"block_M", "block_N", "block_K", "num_stages", "threads", "persistent"}


def _sm_version(target):
    match = re.fullmatch(r"sm_(\d+)[a-z]?", str(target.get("arch", "")))
    return int(match.group(1)) if match else -1


def support_reason(workload, device):
    if _sm_version(device.target) < 100:
        return "the grouped MXFP8 Carver model requires a Blackwell CUDA target"
    from experiments.grouped_gemm.spaces import support_reason as kernel_support_reason

    return kernel_support_reason(workload)


def _model_config(workload, config, arch):
    """Estimate one fixed native schedule without consulting measurements."""
    from experiments.grouped_gemm.spaces import BLOCK_K, BLOCK_M, BLOCK_N, NUM_STAGES

    p = workload.parameters
    fixed = (
        set(config) == _CONFIG_KEYS
        and config["block_M"] == BLOCK_M
        and config["block_N"] == BLOCK_N
        and config["block_K"] == BLOCK_K
        and config["num_stages"] == NUM_STAGES
        and (config["threads"], config["persistent"]) in ((128, False), (256, True))
    )
    cluster_size = 2
    max_m_blocks = math.ceil(max(p["batch_sizes"]) / BLOCK_M)
    m_clusters = math.ceil(max_m_blocks / cluster_size)
    n_tiles = math.ceil(p["n"] / BLOCK_N)
    total_cluster_tiles = len(p["batch_sizes"]) * n_tiles * m_clusters
    resident_clusters = max(1, arch.compute_max_core // cluster_size)
    cluster_waves = math.ceil(total_cluster_tiles / resident_clusters)

    k_iterations = math.ceil(p["k"] / BLOCK_K)
    scale_load_period = 128 * 4 // BLOCK_K
    scale_loads = math.ceil(k_iterations / scale_load_period)
    input_bytes_per_cluster_k = cluster_size * BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K
    scale_bytes_per_cluster_load = (cluster_size * BLOCK_M + cluster_size * BLOCK_N) * 4
    output_bytes_per_cluster = cluster_size * BLOCK_M * BLOCK_N * 2
    traffic_bytes_per_cluster = (
        k_iterations * input_bytes_per_cluster_k
        + scale_loads * scale_bytes_per_cluster_load
        + output_bytes_per_cluster
    )
    traffic_bytes = total_cluster_tiles * traffic_bytes_per_cluster

    launched_ctas = arch.compute_max_core if config.get("persistent") else cluster_size * total_cluster_tiles
    dispatch_bytes_per_cta = arch.transaction_size[-1]
    score = float(traffic_bytes + launched_ctas * dispatch_bytes_per_cta)

    staged_input_bytes = NUM_STAGES * (
        BLOCK_M * BLOCK_K
        + (BLOCK_N // cluster_size) * BLOCK_K
        + BLOCK_M * 4
        + BLOCK_N * 4
    )
    store_columns = 64 if config.get("persistent") else BLOCK_N
    output_shared_bytes = BLOCK_M * store_columns * 2
    shared_bytes = max(staged_input_bytes, output_shared_bytes)
    valid = fixed and shared_bytes <= arch.max_smem_usage and config.get("threads", 0) <= 1024
    return dict(
        valid=valid,
        persistent=bool(config.get("persistent")),
        total_cluster_tiles=total_cluster_tiles,
        resident_clusters=resident_clusters,
        cluster_waves=cluster_waves,
        k_iterations=k_iterations,
        scale_loads=scale_loads,
        traffic_bytes_per_cluster=traffic_bytes_per_cluster,
        traffic_bytes=traffic_bytes,
        launched_ctas=launched_ctas,
        dispatch_bytes_per_cta=dispatch_bytes_per_cta,
        shared_bytes=shared_bytes,
        score=score if valid else None,
    )


def carver_rank(workload, device, configs, top_k):
    reason = support_reason(workload, device)
    if reason:
        raise ValueError(reason)
    from tilelang import tvm
    from tilelang.tiletune.ranking import rank_records, select_top_k
    from experiments.common.carver import _architecture, workload_template

    arch = _architecture(device.target)
    template = workload_template(workload, configs, arch=arch)
    records = []
    for index, config in enumerate(configs):
        model = _model_config(workload, config, arch)
        score = model.pop("score")
        valid = model["valid"]
        records.append(
            dict(
                index=index,
                config=dict(config),
                status="analyzed" if valid else "model_rejected",
                tile_cost=dict(score=score, ranking_metric="carver_grouped_mxfp8_bytes"),
                model=model,
            )
        )
    ranking = rank_records(records)
    selected = select_top_k(ranking, top_k)
    for record in records:
        record["selected"] = record["index"] in selected
        if record["status"] == "analyzed" and not record["selected"]:
            record["status"] = "not_selected"
    return dict(
        model="carver_grouped_mxfp8_v1",
        model_target=str(arch.target),
        compile_target=str(tvm.target.Target(device.target)),
        template=type(template).__name__,
        formula="total_tile_traffic_bytes + launched_ctas * architecture_transaction_bytes",
        ranking=ranking,
        configs=records,
        selection=dict(
            requested_k=top_k,
            selected_indices=selected,
            selected_count=len(selected),
            shortfall=max(0, top_k - len(selected)),
        ),
        metric="carver_grouped_mxfp8_bytes",
        score_units="byte-equivalents",
        assumptions=[
            "The native fixed 128x256x128 schedule executes padded two-CTA cluster tiles.",
            "Both schedules have the same modeled operand, scale, and output traffic per cluster tile.",
            "MXFP8 scale words are loaded every four K tiles by both CTAs, matching the native kernel.",
            "The persistent schedule launches one CTA per SM; the tiled schedule launches two CTAs per cluster tile.",
            "One architecture transaction represents the scheduling cost of each launched CTA.",
            "No compiled latency or oracle measurement enters the analytical ranking.",
        ],
    )
