"""Evaluate the repository's legacy Carver policy on an explicit GEMM grid."""

import json
from pathlib import Path

from tilelang.carver.arch import CUDA
from tilelang.carver.matmul_analysis import get_tensorized_func_and_tags
from tilelang.carver.roller.policy import TensorCorePolicy
from tilelang.carver.template import MatmulTemplate
from tilelang.tiletune.ranking import rank_records, select_top_k
from tvm.target import Target


def model_target(target):
    """The old model accepts sm_90; kernels still compile for the actual sm_90a."""
    target = dict(Target(target).export())
    if target.get("arch") == "sm_90a":
        target["arch"] = "sm_90"
    return Target(target)


def rank_configs(configs, *, m, n, k, dtype, target, top_k):
    baseline = json.loads(Path(__file__).with_name("carver_baseline.json").read_text())
    arch = CUDA(model_target(target))
    template = MatmulTemplate(M=m, N=n, K=k, trans_B=True, in_dtype=dtype, out_dtype=dtype, accum_dtype="float32")
    func, tags = get_tensorized_func_and_tags(template.equivalent_function(), arch.target, allow_gemv=True)
    if func is None or not tags:
        raise ValueError("Legacy Carver cannot tensorize this GEMM; no fallback cost model is used")
    policies = {
        stages: TensorCorePolicy.from_prim_func(func, arch, {**tags, "pipeline_stage": max(1, stages)})
        for stages in sorted({cfg["num_stages"] for cfg in configs})
    }
    records = []
    for index, config in enumerate(configs):
        policy = policies[config["num_stages"]]
        steps = {node: {axis.var.name: config["block_K"] for axis in node.raxis} for node in policy.ordered_nodes}
        td = policy.compute_tile_dict([config["block_M"], config["block_N"]], steps)
        valid = td.valid and policy.check_tile_shape_isvalid(td)
        if valid:
            valid = all(policy._assign_block_size(node, td, config["thread_num"]) is not None for node in policy.ordered_nodes)
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
        model="legacy_carver_common_grid",
        baseline=baseline,
        model_target=str(arch.target),
        compile_target=str(Target(target)),
        formula="(traffic_bytes + 1) * num_wave",
        ranking=ranking,
        configs=records,
        selection=dict(requested_k=top_k, selected_indices=selected, selected_count=len(selected), shortfall=top_k - len(selected)),
        assumptions=[
            "Uses the repository's legacy Carver policy; baseline metadata records the restored upstream snapshot.",
            "sm_90a is spelled sm_90 for the legacy model only; every method compiles the same kernel for the actual target.",
            "Pipeline stages 0 and 1 both use one shared-memory copy in the legacy policy.",
            "Thread count is checked for policy feasibility; the cost retains Carver's original occupancy estimate.",
            "Rasterization is not distinguished by this score; equal scores retain original grid order.",
            "Candidate generation and reduction-step expansion are disabled for this common-grid comparison.",
        ],
    )
