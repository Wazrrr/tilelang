"""Family-specific adapters from experiment grids to Carver templates."""

import math


def workload_template(workload, configs=None, *, arch=None):
    """Build the canonical Carver template for an experiment workload."""
    from tilelang.carver.template import (
        FP8MatmulTemplate,
        FlashAttentionTemplate,
        GroupedMXFP8MatmulTemplate,
        KDAChunkTemplate,
        MatmulTemplate,
    )

    p = workload.parameters
    common = {"_arch": arch} if arch is not None else {}
    if workload.op == "gemm":
        return MatmulTemplate(
            M=p["m"],
            N=p["n"],
            K=p["k"],
            trans_A=p.get("transpose_a", False),
            trans_B=p.get("transpose_b", False),
            in_dtype=workload.dtype,
            out_dtype=workload.dtype,
            accum_dtype="float32",
            with_bias=p.get("epilogue", "none") != "none",
            **common,
        )
    if workload.op == "gemm_fp8":
        return FP8MatmulTemplate(
            M=p["m"],
            N=p["n"],
            K=p["k"],
            trans_A=p.get("transpose_a", False),
            trans_B=p.get("transpose_b", False),
            kernel_dtype=workload.dtype,
            **common,
        )
    if workload.op == "attention":
        return FlashAttentionTemplate(
            batch_size=p["batch"],
            num_heads=p["heads"],
            seq_length=p["sequence"],
            seq_kv_length=p["sequence"],
            head_dim=p["dim"],
            is_causal=p.get("causal", False),
            in_dtype=workload.dtype,
            out_dtype=workload.dtype,
            accum_dtype="float32",
            **common,
        )
    if workload.op == "grouped_gemm":
        block_sizes = {config["block_M"] for config in configs or []}
        if not block_sizes:
            from experiments.grouped_gemm.spaces import BLOCK_M

            block_sizes = {BLOCK_M}
        if len(block_sizes) != 1:
            raise ValueError("one grouped-GEMM Carver template requires one fixed block_M")
        return GroupedMXFP8MatmulTemplate(
            batch_sizes=list(p["batch_sizes"]),
            block_m=block_sizes.pop(),
            N=p["n"],
            K=p["k"],
            trans_B=p.get("transpose_b", False),
            kernel_dtype=workload.dtype,
            scale_granularity_k=128,
            cluster_size=2,
            **common,
        )
    if workload.op == "kda_chunk_o":
        return KDAChunkTemplate(
            batch_size=p["batch"],
            num_heads=p["heads"],
            sequence=p["sequence"],
            key_dim=p["dim"],
            value_dim=p["value_dim"],
            chunk_size=p["chunk_size"],
            in_dtype=workload.dtype,
            out_dtype=workload.dtype,
            accum_dtype="float32",
            **common,
        )
    raise ValueError(f"No Carver template for experiment operation {workload.op!r}")


def _rank_records(configs, top_k, *, arch, template, evaluate):
    from tilelang.tiletune.ranking import rank_records, select_top_k

    records = []
    for index, config in enumerate(configs):
        model = evaluate(config)
        valid = model.pop("valid")
        score = float((model["traffic_bytes_per_cta"] + 1) * model["waves"]) if valid else None
        records.append(
            dict(
                index=index,
                config=dict(config),
                status="analyzed" if valid else "model_rejected",
                tile_cost=dict(score=score, ranking_metric="carver_traffic_waves"),
                model=dict(valid=valid, **model),
            )
        )
    ranking = rank_records(records)
    selected = select_top_k(ranking, top_k)
    for record in records:
        record["selected"] = record["index"] in selected
        if record["status"] == "analyzed" and not record["selected"]:
            record["status"] = "not_selected"
    return dict(
        model="legacy_carver_common_grid",
        model_target=str(arch.target),
        template=type(template).__name__,
        formula="(traffic_bytes_per_cta + 1) * num_waves",
        ranking=ranking,
        configs=records,
        selection=dict(requested_k=top_k, selected_indices=selected, selected_count=len(selected), shortfall=top_k - len(selected)),
        metric="carver_traffic_waves",
        score_units="byte-waves",
        assumptions=[
            "The family template preserves the kernel's CTA domain, tile traffic, shared storage, and accumulator footprint.",
            "Candidate generation is disabled; Carver scores every configuration from the shared experiment pool.",
            "SM100 uses Carver's SM90 tensorization vocabulary while retaining the visible device's SM and storage capacities.",
            "Rasterization is not distinguished by the legacy score; equal scores retain original grid order.",
        ],
    )


def _architecture(target):
    from tilelang.carver.arch import CUDA
    from experiments.gemm.carver import model_target

    return CUDA(model_target(target))


def _occupancy(arch, *, grid_blocks, shared_bytes, register_words, threads):
    # ``smem_cap`` is the static allocation ceiling returned by the driver;
    # TileLang opts these kernels into the larger dynamic per-block limit.
    valid = shared_bytes <= arch.max_smem_usage and register_words <= arch.reg_cap and threads <= 1024
    if not valid:
        return False, 0, 0
    blocks_per_sm = max(
        1,
        min(
            arch.max_smem_usage // max(shared_bytes, 1),
            arch.reg_cap // max(register_words, 1),
            arch.sm_partition,
        ),
    )
    return True, blocks_per_sm, math.ceil(grid_blocks / max(1, blocks_per_sm * arch.compute_max_core))


def _full_row_gemm_supported(m, n, threads):
    """Mirror the example's CUDA FullRow warp and fragment-layout contract."""
    warps = threads // 32
    partition = any(
        row_warps * column_warps == warps and m % (16 * row_warps) == 0 and n % (8 * column_warps) == 0
        for row_warps in range(1, warps + 1)
        for column_warps in range(1, warps + 1)
    )
    # The attention recurrence casts its score accumulator in a parallel loop.
    # Its inferred fragment layout requires one complete row group per thread
    # cohort: 64 rows for 128 threads and 128 rows for 256 threads.
    return partition and m % (threads // 2) == 0


def attention_rank(workload, device, configs, top_k):
    p, element_bytes = workload.parameters, 2
    batch, heads, sequence, dim = (p[key] for key in ("batch", "heads", "sequence", "dim"))
    arch = _architecture(device.target)
    # Keep the semantic graph as the source of workload identity. The grid
    # adapter evaluates the online tiled recurrence without materializing its
    # full score matrix.
    template = workload_template(workload, configs, arch=arch)

    def evaluate(c):
        bm, bn, depth = c["block_M"], c["block_N"], max(1, c["num_stages"])
        grid = batch * heads * math.ceil(sequence / bm)
        iterations = math.ceil(sequence / bn)
        if p.get("causal", False):
            # Exact total visits across query CTAs. The maximum CTA remains the
            # service-time bound used for waves.
            visits = sum(min(iterations, math.ceil((block + 1) * bm / bn)) for block in range(math.ceil(sequence / bm)))
            average_iterations = visits / math.ceil(sequence / bm)
        else:
            average_iterations = iterations
        traffic = element_bytes * (2 * bm * dim + average_iterations * 2 * bn * dim)
        shared = element_bytes * (2 * bm * dim + 2 * bn * dim * depth)
        register_words = math.ceil((bm * bn * 6 + bm * dim * 4 + bm * 5 * 4) / 4)
        valid, blocks, waves = _occupancy(
            arch, grid_blocks=grid, shared_bytes=shared, register_words=register_words, threads=c["threads"]
        )
        valid = (
            valid
            and _full_row_gemm_supported(bm, bn, c["threads"])
            and _full_row_gemm_supported(bm, dim, c["threads"])
        )
        return dict(
            valid=valid,
            traffic_bytes_per_cta=traffic,
            shared_bytes=shared,
            register_words=register_words,
            grid_blocks=grid,
            blocks_per_sm=blocks,
            waves=waves,
            loop_iterations=average_iterations,
        )

    return _rank_records(configs, top_k, arch=arch, template=template, evaluate=evaluate)


def kda_rank(workload, device, configs, top_k):
    p = workload.parameters
    batch, heads, sequence, dk, dv, chunk = (p[key] for key in ("batch", "heads", "sequence", "dim", "value_dim", "chunk_size"))
    element_bytes = 2
    arch = _architecture(device.target)
    template = workload_template(workload, configs, arch=arch)

    def evaluate(c):
        bdk, bdv, depth = c["block_DK"], c["block_DV"], max(1, c["num_stages"])
        grid = batch * heads * (sequence // chunk) * math.ceil(dv / bdv)
        iterations = math.ceil(dk / bdk)
        repeated = chunk * bdk * (2 * element_bytes + 4) + bdk * bdv * element_bytes
        once = (chunk * bdv + chunk * chunk + chunk * bdv) * element_bytes
        traffic = iterations * repeated + once
        shared = depth * repeated + once
        register_words = chunk * bdv
        valid, blocks, waves = _occupancy(
            arch, grid_blocks=grid, shared_bytes=shared, register_words=register_words, threads=c["threads"]
        )
        return dict(
            valid=valid,
            traffic_bytes_per_cta=traffic,
            shared_bytes=shared,
            register_words=register_words,
            grid_blocks=grid,
            blocks_per_sm=blocks,
            waves=waves,
            loop_iterations=iterations,
        )

    return _rank_records(configs, top_k, arch=arch, template=template, evaluate=evaluate)
