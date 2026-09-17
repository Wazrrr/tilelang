"""Simple analytical model for the fixed SM100 MXFP8 schedules."""

import math
import re


def support_reason(workload, device):
    match = re.fullmatch(r"sm_(\d+)[af]?", str(device.target.get("arch", "")))
    if device.target.get("kind") != "cuda" or not match or int(match[1]) < 100:
        return "the TCGen05 FP8 Carver model requires a Blackwell CUDA target"
    from experiments.gemm_fp8.spaces import support_reason as kernel_support_reason

    return kernel_support_reason(workload, device)


def carver_rank(workload, device, configs, top_k):
    reason = support_reason(workload, device)
    if reason:
        raise ValueError(reason)
    from tilelang import tvm
    from tilelang.tiletune.ranking import rank_records, select_top_k
    from experiments.common.carver import _architecture, workload_template
    from experiments.gemm_fp8.spaces import get_configs

    arch = _architecture(device.target)
    template = workload_template(workload, configs, arch=arch)
    p = workload.parameters
    cluster_tiles = math.ceil(p["m"] / 256) * math.ceil(p["n"] / 256)
    resident_clusters = max(1, arch.compute_max_core // 2)
    cluster_waves = math.ceil(cluster_tiles / resident_clusters)
    k_blocks = math.ceil(p["k"] / 128)
    operand_bytes = cluster_tiles * k_blocks * (256 * 128 + 256 * 128)
    scale_bytes = cluster_tiles * math.ceil(k_blocks / 4) * (256 + 256) * 4
    output_bytes = cluster_tiles * 256 * 256 * 2
    traffic_bytes = operand_bytes + scale_bytes + output_bytes
    declared = get_configs()

    records = []
    for index, config in enumerate(configs):
        valid = config in declared
        persistent = config.get("implementation", "").endswith("persistent")
        launched_ctas = arch.compute_max_core if persistent else 2 * cluster_tiles
        store_passes = 1 if not config.get("use_tma_store", True) else 256 // config["store_block_N"]
        dispatch_cost = launched_ctas * arch.transaction_size[-1]
        score = float(traffic_bytes * cluster_waves + dispatch_cost + store_passes * output_bytes) if valid else None
        records.append(
            dict(
                index=index,
                config=dict(config),
                status="analyzed" if valid else "model_rejected",
                tile_cost=dict(score=score, ranking_metric="carver_sm100_mxfp8_bytes"),
                model=dict(
                    valid=valid,
                    persistent=persistent,
                    cluster_tiles=cluster_tiles,
                    resident_clusters=resident_clusters,
                    cluster_waves=cluster_waves,
                    k_blocks=k_blocks,
                    traffic_bytes=traffic_bytes,
                    launched_ctas=launched_ctas,
                    store_passes=store_passes,
                ),
            )
        )
    ranking = rank_records(records)
    selected = select_top_k(ranking, top_k)
    for record in records:
        record["selected"] = record["index"] in selected
        if record["status"] == "analyzed" and not record["selected"]:
            record["status"] = "not_selected"
    return dict(
        model="carver_sm100_mxfp8_v1",
        model_target=str(arch.target),
        compile_target=str(tvm.target.Target(device.target)),
        template=type(template).__name__,
        formula="traffic_bytes * cluster_waves + dispatch_bytes + store_passes * output_bytes",
        ranking=ranking,
        configs=records,
        selection=dict(
            requested_k=top_k,
            selected_indices=selected,
            selected_count=len(selected),
            shortfall=max(0, top_k - len(selected)),
        ),
        metric="carver_sm100_mxfp8_bytes",
        score_units="byte-equivalents",
        assumptions=[
            "The native kernel uses fixed 128x256x128 two-CTA TCGen05 tiles and packed UE8M0 scales.",
            "Persistent swizzle sizes affect scheduling order but not modeled data traffic.",
            "No measured latency or oracle outcome enters the ranking.",
        ],
    )
