"""Family-specific adapters from experiment grids to Carver templates."""

import math


def workload_template(workload, configs=None, *, arch=None):
    """Build the canonical Carver template for an experiment workload."""
    from tilelang.carver.template import (
        FP8MatmulTemplate,
        FlashAttentionTemplate,
        GroupedMatmulTemplate,
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
        return GroupedMatmulTemplate(
            batch_sizes=list(p["batch_sizes"]),
            block_m=block_sizes.pop(),
            N=p["n"],
            K=p["k"],
            trans_B=p.get("transpose_b", False),
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
    selected = select_top_k(ranking, top_k, include_ties=False)
    for record in records:
        record["selected"] = record["index"] in selected
        if record["status"] == "analyzed" and not record["selected"]:
            record["status"] = "not_selected"
    return dict(
        model="legacy_carver_common_grid",
        model_target=str(arch.target),
        template=type(template).__name__ if template is not None else None,
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
    from tilelang.carver.roller.policy.common import coalesced_tensor_shape

    p, element_bytes = workload.parameters, 2
    batch, heads, sequence, dim = (p[key] for key in ("batch", "heads", "sequence", "dim"))
    arch = _architecture(device.target)
    # Keep the semantic graph as the source of workload identity. The grid
    # adapter evaluates the online tiled recurrence without materializing its
    # full score matrix.
    template = workload_template(workload, configs, arch=arch)
    shape = [batch, sequence, heads, dim]

    def transfer(rows, direction):
        return coalesced_tensor_shape([1, rows, 1, dim], shape, arch.transaction_size[direction] // element_bytes) * element_bytes

    def evaluate(c):
        bm, bn, depth = c["block_M"], c["block_N"], max(1, c["num_stages"])
        grid = batch * heads * math.ceil(sequence / bm)
        iterations = math.ceil(sequence / bn)
        query_blocks = math.ceil(sequence / bm)
        visits = [
            math.ceil(min((block + 1) * bm, sequence) / bn) if p.get("causal", False) else iterations for block in range(query_blocks)
        ]
        average_iterations = sum(visits) / query_blocks
        once = (
            sum(transfer(min(bm, sequence - block * bm), direction) for block in range(query_blocks) for direction in (0, 1)) / query_blocks
        )
        key_value = 0.0
        for count in visits:
            full, tail = divmod(min(count * bn, sequence), bn)
            key_value += 2 * (full * transfer(bn, 1) + transfer(tail, 1)) / query_blocks
        traffic = once + key_value
        shared = element_bytes * (2 * bm * dim + 2 * bn * dim * depth)
        register_words = math.ceil((bm * bn * 6 + bm * dim * 4 + bm * 5 * 4) / 4)
        valid, blocks, waves = _occupancy(arch, grid_blocks=grid, shared_bytes=shared, register_words=register_words, threads=c["threads"])
        valid = valid and _full_row_gemm_supported(bm, bn, c["threads"]) and _full_row_gemm_supported(bm, dim, c["threads"])
        return dict(
            valid=valid,
            traffic_bytes_per_cta=traffic,
            shared_bytes=shared,
            register_words=register_words,
            grid_blocks=grid,
            blocks_per_sm=blocks,
            waves=waves,
            loop_iterations=average_iterations,
            once_bytes=once,
            key_value_bytes=key_value,
            query_blocks=query_blocks,
            traffic_basis="mean per-CTA BSHD transfers, clipped tails and Carver transaction sizes",
        )

    return _rank_records(configs, top_k, arch=arch, template=template, evaluate=evaluate)


def kda_intra_rank(workload, device, configs, top_k):
    """Carver traffic/wave model for the token-parallel KDA intra kernel.

    The kernel issues no tensor-core MMA, so this adapter scores the kernel's
    own CTA domain and memory ledger instead of a tensorized GEMM. Each
    ``(token, head-block)`` CTA reads its Q/K/gate/beta row once, then streams
    K/gate for every causal predecessor token inside the chunk to form the
    gated ``Aqk = Q K^T`` and ``Akk = (beta K) K^T`` coefficient tiles.
    """
    from tilelang.carver.roller.policy.common import coalesced_tensor_shape

    p = workload.parameters
    batch, heads, sequence, dim = (p[key] for key in ("batch", "heads", "sequence", "dim"))
    chunk, sub_chunk = p["chunk_size"], p["sub_chunk_size"]
    arch = _architecture(device.target)
    in_bytes, gate_bytes, out_bytes = 2, 4, 2
    shape = [batch, sequence, heads, dim]

    def transfer(rows, columns, direction, element_bytes):
        transaction = arch.transaction_size[direction] // element_bytes
        return coalesced_tensor_shape([1, 1, rows, columns], shape, transaction) * element_bytes

    # Mean causal predecessors visited per token inside one sub-chunk.
    average_iterations = (1 + sub_chunk) / 2

    def evaluate(c):
        block_h, block_dk, depth = c["block_H"], c["block_DK"], max(1, c["num_stages"])
        head_blocks = math.ceil(heads / block_h)
        grid = batch * sequence * head_blocks
        tiles = dim // block_dk
        streamed = average_iterations * tiles * (
            transfer(block_h, block_dk, 0, in_bytes) + transfer(block_h, block_dk, 0, gate_bytes)
        )
        traffic = (
            transfer(block_h, dim, 0, in_bytes)  # Q
            + transfer(block_h, dim, 0, in_bytes)  # K
            + transfer(block_h, dim, 0, gate_bytes)  # gate
            + block_h * in_bytes  # beta
            + streamed
            + transfer(block_h, chunk, 1, out_bytes)  # Aqk
            + transfer(block_h, sub_chunk, 1, out_bytes)  # Akk
        )
        shared = (
            block_h * dim * (2 * in_bytes + gate_bytes)
            + block_h * in_bytes
            + depth * block_h * block_dk * (in_bytes + gate_bytes)
            + block_h * chunk * out_bytes
            + block_h * sub_chunk * out_bytes
        )
        register_words = math.ceil(3 * block_h * block_dk + 2 * block_h)
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
            loop_iterations=average_iterations,
            traffic_basis="mean per-CTA gated QK/KK causal transfers with Carver transaction sizes",
        )

    report = _rank_records(configs, top_k, arch=arch, template=None, evaluate=evaluate)
    report["template"] = "kda_intra_token_parallel"
    report["assumptions"].append(
        "KDA issues no tensor-core MMA: the score uses its CTA domain and gated QK/KK memory ledger, not a tensorized GEMM."
    )
    return report
