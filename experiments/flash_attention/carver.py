"""Evaluate Carver's original two-GEMM attention graph on the frozen FA pool."""


def attention_template(arch, *, batch, heads, sequence, dim, causal, dtype):
    from dataclasses import dataclass, field
    from tilelang.carver.template import FlashAttentionTemplate

    # The base constructor otherwise infers a target before with_arch can run.
    # Only expose constructor injection; inherit the original graph unchanged.
    @dataclass
    class ExplicitArchAttention(FlashAttentionTemplate):
        _arch: object = field(default=None, repr=False)

    return ExplicitArchAttention(
        batch_size=batch,
        num_heads=heads,
        seq_length=sequence,
        seq_kv_length=sequence,
        head_dim=dim,
        is_causal=causal,
        in_dtype=dtype,
        out_dtype=dtype,
        accum_dtype="float32",
        _arch=arch,
    )


def rank_configs(configs, *, batch, heads, sequence, dim, causal, dtype, target, top_k):
    from tilelang import tvm
    from tilelang.carver.arch import CUDA
    from tilelang.carver.roller.policy import TensorCorePolicy
    from tilelang.tiletune.ranking import rank_records, select_top_k
    from experiments.gemm.carver import model_target

    arch = CUDA(model_target(target))
    template = attention_template(arch, batch=batch, heads=heads, sequence=sequence, dim=dim, causal=causal, dtype=dtype)
    policy = TensorCorePolicy.from_output_nodes(template.output_nodes, arch)
    qk, pv = policy.ordered_nodes
    qk_step = policy._assign_reduce_step(qk)
    records = []
    for index, config in enumerate(configs):
        # Configure the existing stage parameter, not the policy's equations.
        policy.pipeline_stage = max(1, config["num_stages"])
        steps = {qk: dict(qk_step), pv: {axis.var.name: config["block_N"] for axis in pv.raxis}}
        td = policy.compute_tile_dict([1, config["block_M"], dim], steps)
        reason = None
        if not td.valid:
            reason = "shared_memory" if td.smem_cost > arch.smem_cap else "registers"
        elif not policy.check_tile_shape_isvalid(td):
            reason = "tile_shape"
        elif any(policy._assign_block_size(node, td, config["threads"]) is None for node in policy.ordered_nodes):
            reason = "thread_assignment"
        valid = reason is None
        # Exactly DefaultPolicy.dfs_smem_tile's priority. No candidate expansion
        # or replacement for rejected candidates; only the supplied pool is used.
        score = float((td.traffic + 1) * td.num_wave) if valid else None
        records.append(
            dict(
                index=index,
                config=dict(config),
                status="analyzed" if valid else "model_rejected",
                tile_cost=dict(score=score, ranking_metric="carver_traffic_waves"),
                model=dict(
                    valid=valid,
                    rejection_reason=reason,
                    traffic_bytes=float(td.traffic),
                    shared_bytes=int(td.smem_cost),
                    waves=int(td.num_wave) if td.valid else None,
                    blocks_per_sm=int(td.block_per_SM) if td.valid else None,
                    pipeline_stage=policy.pipeline_stage,
                    qk_tile=[int(v) for v in td.get_tile(qk)],
                    pv_tile=[int(v) for v in td.get_tile(pv)],
                    qk_reduction={k: int(v) for k, v in steps[qk].items()},
                    pv_reduction={k: int(v) for k, v in steps[pv].items()},
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
        model="legacy_carver_attention_common_grid",
        model_target=str(arch.target),
        compile_target=str(tvm.target.Target(target)),
        formula="(traffic_bytes + 1) * num_wave",
        model_limits=dict(
            shared_bytes_per_block=int(arch.smem_cap), shared_bytes_per_sm=int(arch.max_smem_usage), registers=int(arch.reg_cap)
        ),
        ranking=ranking,
        configs=records,
        selection=dict(requested_k=top_k, selected_indices=selected, selected_count=len(selected), shortfall=max(0, top_k - len(selected))),
        assumptions=[
            "Uses the unchanged FlashAttentionTemplate and TensorCorePolicy; measures the authoritative BSHD example kernel.",
            "The original graph contains QK and PV only: no softmax, scaling, causal mask, or causal work reduction.",
            "Carver models both matmuls with transposed second operands; the measured example retains its original BSHD layouts.",
            "Output tile is [1, block_M, head_dim]; block_N sets the PV reduction step; QK uses Carver's default reduction step.",
            "Carver propagates a full sequence-wide QK tile, not the example's streamed KV tile. Its memory limits are unchanged.",
            "Pipeline stages 0 and 1 both use one shared-memory copy in the legacy policy.",
            "Thread count is checked for policy feasibility; the score retains Carver's original occupancy estimate.",
            "sm_90a is normalized to sm_90 for the model only; kernel compilation uses the actual target.",
            "Candidate generation and reduction-step expansion are disabled; equal scores retain original pool order.",
        ],
    )


def carver_support_reason(workload, device):
    from experiments.common.baselines import carver_target_support_reason

    reason = carver_target_support_reason(device.target)
    if reason:
        return reason
    if workload.op != "attention" or workload.dtype not in ("float16", "bfloat16"):
        return "the Carver attention adapter supports FP16/BF16 attention only"
    from experiments.common.spec import configurations

    keys = {"block_M", "block_N", "num_stages", "threads"}
    if any(set(c) != keys for c in configurations(workload, device)):
        return "Carver requires the attention example's configuration schema"
    return None


def carver_rank(workload, device, configs, top_k):
    reason = carver_support_reason(workload, device)
    if reason:
        raise ValueError(reason)
    report = rank_configs(configs, **(dict(causal=False) | workload.parameters), dtype=workload.dtype, target=device.target, top_k=top_k)
    report.update(metric="carver_traffic_waves", score_units="byte-waves")
    return report
