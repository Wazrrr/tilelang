"""Evaluate the repository's legacy Carver policy on an explicit GEMM grid."""


def model_target(target):
    """The old model accepts sm_90; kernels still compile for the actual sm_90a."""
    from tilelang import tvm

    Target = tvm.target.Target

    target = dict(Target(target).export())
    if target.get("arch") in ("sm_90a", "sm_100a"):
        target["arch"] = "sm_90"
    return Target(target)


def rank_configs(
    configs,
    *,
    m,
    n,
    k,
    dtype,
    target,
    top_k,
    transpose_a=False,
    transpose_b=True,
    template=None,
    thread_key="thread_num",
):
    from tilelang import tvm

    Target = tvm.target.Target
    from tilelang.carver.arch import CUDA
    from tilelang.carver.matmul_analysis import get_tensorized_func_and_tags
    from tilelang.carver.roller.policy import TensorCorePolicy
    from tilelang.carver.template import MatmulTemplate
    from tilelang.tiletune.ranking import rank_records, select_top_k

    compile_target = Target(target)
    arch = CUDA(model_target(target))
    template = template or MatmulTemplate(
        M=m,
        N=n,
        K=k,
        trans_A=transpose_a,
        trans_B=transpose_b,
        in_dtype=dtype,
        out_dtype=dtype,
        accum_dtype="float32",
        _arch=arch,
    )
    func, tags = get_tensorized_func_and_tags(template.equivalent_function(), arch.target, allow_gemv=True)
    if func is None or not tags:
        raise ValueError("Legacy Carver cannot tensorize this GEMM; no fallback cost model is used")
    policies = {
        stages: TensorCorePolicy.from_prim_func(func, arch, {**tags, "pipeline_stage": max(1, stages)})
        for stages in sorted({cfg["num_stages"] for cfg in configs})
    }
    target_arch = str(compile_target.attrs.get("arch", ""))
    tcgen05_meta = tvm.get_global_func("tl.get_tcgen5_mma_meta") if target_arch.startswith("sm_100") else None
    records = []
    for index, config in enumerate(configs):
        policy = policies[config["num_stages"]]
        steps = {node: {axis.var.name: config["block_K"] for axis in node.raxis} for node in policy.ordered_nodes}
        td = policy.compute_tile_dict([config["block_M"], config["block_N"]], steps)
        native_legal = (
            bool(
                tcgen05_meta(
                    config["block_M"],
                    config["block_N"],
                    config["block_K"],
                    dtype,
                    "float32",
                    True,
                    False,
                )
            )
            if tcgen05_meta is not None
            else True
        )
        valid = native_legal and td.valid and policy.check_tile_shape_isvalid(td)
        if valid:
            valid = all(policy._assign_block_size(node, td, config[thread_key]) is not None for node in policy.ordered_nodes)
        # Exactly DefaultPolicy.dfs_smem_tile's priority, evaluated at the supplied
        # reduction step. Do not expand tiles/steps or substitute TileTune estimates.
        score = float((td.traffic + 1) * td.num_wave) if valid else None
        records.append(
            dict(
                index=index,
                config=dict(config),
                status="analyzed" if valid else "model_rejected",
                tile_cost=dict(score=score, ranking_metric="carver_traffic_waves"),
                model=dict(
                    valid=bool(valid),
                    traffic_bytes=float(td.traffic),
                    shared_bytes=int(td.smem_cost),
                    waves=int(td.num_wave) if td.valid else None,
                    blocks_per_sm=int(td.block_per_SM) if td.valid else None,
                    native_lowering_legal=bool(native_legal),
                ),
            )
        )
    ranking = rank_records(records)
    # A Carver rejection is an unsupported candidate, not an unscored
    # exploration candidate.  Never fill a shortfall with those records.
    selected = select_top_k(ranking, top_k, include_unknown=False)
    for record in records:
        record["selected"] = record["index"] in selected
        if record["status"] == "analyzed" and not record["selected"]:
            record["status"] = "not_selected"
    return dict(
        model="legacy_carver_common_grid",
        model_target=str(arch.target),
        compile_target=str(Target(target)),
        template=type(template).__name__,
        formula="(traffic_bytes + 1) * num_wave",
        ranking=ranking,
        configs=records,
        selection=dict(requested_k=top_k, selected_indices=selected, selected_count=len(selected), shortfall=top_k - len(selected)),
        assumptions=[
            "Uses the repository's legacy Carver policy without experimental WGMMA/WS model extensions.",
            "sm_90a is spelled sm_90 for the legacy model only; every method compiles the same kernel for the actual target.",
            "Pipeline stages 0 and 1 both use one shared-memory copy in the legacy policy.",
            "Thread count is checked for policy feasibility; the cost retains Carver's original occupancy estimate.",
            "Rasterization is not distinguished by this score; equal scores retain original grid order.",
            "Candidate generation and reduction-step expansion are disabled for this common-grid comparison.",
            "On SM100, candidates unsupported by the native one-CTA TCGen05 lowering are rejected before ranking.",
        ],
    )


def carver_support_reason(workload, device):
    if device.target["kind"] != "cuda":
        return "the existing Carver comparison adapter requires CUDA"
    if workload.op != "gemm" or workload.dtype not in ("float16", "bfloat16"):
        return "the existing Carver comparison adapter supports FP16/BF16 GEMM only"
    from experiments.gemm.spaces import support_reason

    reason = support_reason(workload)
    if reason:
        return reason
    from experiments.common.spec import configurations

    keys = {"block_M", "block_N", "block_K", "num_stages", "thread_num", "enable_rasteration"}
    if any(set(c) != keys for c in configurations(workload, device)):
        return "Carver requires the example's configuration schema"
    return None


def carver_rank(workload, device, configs, top_k):
    reason = carver_support_reason(workload, device)
    if reason:
        raise ValueError(reason)
    from experiments.common.carver import _architecture, workload_template

    p = workload.parameters
    template = workload_template(workload, configs, arch=_architecture(device.target))
    report = rank_configs(
        configs,
        m=p["m"],
        n=p["n"],
        k=p["k"],
        dtype=workload.dtype,
        target=device.target,
        top_k=top_k,
        transpose_a=p.get("transpose_a", False),
        transpose_b=p.get("transpose_b", False),
        template=template,
    )
    report.update(metric="carver_traffic_waves", score_units="byte-waves")
    return report
