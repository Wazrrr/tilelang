"""Evaluate Carver's template graphs at the experiment's exact schedule knobs.

The original traffic-times-waves priority is retained. For attention, evaluate
one key tile and aggregate the actual number of key tiles for each query block.
This is a Carver traffic model, not a claim of cycle-accurate online execution.
"""

from math import ceil


def rank_graph(workload, device, configs, top_k):
    from tilelang.carver.arch import CUDA
    from tilelang.carver.template import FlashAttentionTemplate, KDAChunkOutputTemplate
    from tilelang.carver.roller.policy import TensorCorePolicy
    from tilelang.tiletune.ranking import rank_records, select_top_k
    from experiments.gemm.carver import model_target

    arch = CUDA(model_target(device.target))
    p = workload.parameters
    policies, records = {}, []
    for index, config in enumerate(configs):
        attention = workload.op == "attention"
        key = (config.get("block_N"), config["num_stages"])
        if key not in policies:
            args = dict(
                batch_size=p["batch"],
                num_heads=p["heads"],
                seq_length=p["sequence"],
                head_dim=p["dim"],
                in_dtype=workload.dtype,
                out_dtype=workload.dtype,
                accum_dtype="float32",
            )
            if attention:
                template = FlashAttentionTemplate(**args, seq_kv_length=config["block_N"], is_causal=p.get("causal", False))
            else:
                template = KDAChunkOutputTemplate(**args, value_dim=p["value_dim"], chunk_size=p["chunk_size"])
            template.with_arch(arch)
            policy = TensorCorePolicy.from_output_nodes(
                template.output_nodes,
                arch,
                {
                    "pipeline_stage": max(1, config["num_stages"]),
                    "allow_partial_tiles": True,
                    "allow_broadcast_recompute": True,
                },
            )
            policies[key] = policy
        policy = policies[key]
        bm = config["block_M"] if attention else p["chunk_size"]
        bn = p["dim"] if attention else config["block_DV"]
        steps = {}
        for node in policy.ordered_nodes:
            if attention:
                step = p["dim"] if node.name == "qk" else config["block_N"]
            else:
                step = config["block_DK"] if node.name == "query_state" else p["chunk_size"]
            steps[node] = {axis.var.name: step for axis in node.raxis}
        model = {}
        try:
            td = policy.compute_tile_dict([1, bm, bn], steps)
            valid = td.valid and policy.check_tile_shape_isvalid(td)
            if valid:
                valid = all(policy._assign_block_size(node, td, config["threads"]) is not None for node in policy.ordered_nodes)
            iterations = 1.0
            if attention:
                query_blocks = ceil(p["sequence"] / bm)
                iterations = (
                    sum(ceil(min((q + 1) * bm, p["sequence"]) / config["block_N"]) for q in range(query_blocks)) / query_blocks
                    if p.get("causal", False)
                    else ceil(p["sequence"] / config["block_N"])
                )
            score = float((td.traffic + 1) * td.num_wave * iterations) if valid else None
            model = dict(
                valid=bool(valid),
                traffic_bytes=float(td.traffic),
                shared_bytes=int(td.smem_cost),
                waves=int(td.num_wave) if td.valid else None,
                key_tile_iterations=iterations,
                stages=[node.name for node in policy.ordered_nodes],
            )
            status = "analyzed" if valid else "model_rejected"
        except (ValueError, AssertionError, IndexError) as error:
            score, status = None, "model_rejected"
            model = dict(valid=False, reason=str(error))
        records.append(
            dict(
                index=index,
                config=dict(config),
                status=status,
                tile_cost=dict(score=score, ranking_metric="carver_traffic_waves"),
                model=model,
            )
        )
    ranking = rank_records(records)
    selected = select_top_k(ranking, top_k)
    for record in records:
        record["selected"] = record["index"] in selected
    return dict(
        configs=records,
        ranking=ranking,
        selection=dict(requested_k=top_k, selected_indices=selected, selected_count=len(selected), shortfall=top_k - len(selected)),
        metric="carver_traffic_waves",
        score_units="byte-waves",
        template="FlashAttentionTemplate" if workload.op == "attention" else "KDAChunkOutputTemplate",
        assumptions=[
            "Carver models fused logical tensor graphs; register ownership and online recurrence cycles are not modeled",
            "attention scores aggregate per-key-tile traffic-waves over the causal or noncausal loop extent",
            "logical batch/head/chunk groups are flattened; the compiled experiment retains the example's BSHD storage",
        ],
    )
