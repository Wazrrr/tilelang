"""Carver's fused gated-state plus causal-local KDA graph."""


def carver_support_reason(workload, device):
    from experiments.common.baselines import carver_target_support_reason

    return carver_target_support_reason(device.target)


def carver_rank(workload, device, configs, top_k):
    from tilelang.carver.arch import CUDA
    from tilelang.carver.template import KDAChunkTemplate
    from tilelang.carver.roller.policy import TensorCorePolicy
    from experiments.gemm.carver import model_target
    from experiments.utils.carver import evaluate_tile, template_report

    reason = carver_support_reason(workload, device)
    if reason:
        raise ValueError(reason)
    arch = CUDA(model_target(device.target))
    template = KDAChunkTemplate(**workload.parameters, in_dtype=workload.dtype, out_dtype=workload.dtype, _arch=arch)
    policy = TensorCorePolicy.from_output_nodes(template.output_nodes, arch)
    state, local = template.stage_nodes
    evaluations = []
    for config in configs:
        policy.pipeline_stage = max(1, config["num_stages"])
        steps = {
            node: {axis.var.name: (config["block_DK"] if node is state else workload.parameters["chunk_size"]) for axis in node.raxis}
            for node in policy.ordered_nodes
        }
        evaluations.append([evaluate_tile(policy, [1, workload.parameters["chunk_size"], config["block_DV"]], steps, config["threads"])])
    return template_report(
        configs,
        evaluations,
        top_k,
        template="carver_kda_chunk",
        assumptions=[
            "Template retains scaling, both input-dtype rounding points, exp2 gates, causal mask, both GEMMs and the output cast.",
            "The two GEMMs feed a fused output node; intermediate state/local results are graph edges, not global transfers.",
            "Uses the original Carver memory/occupancy priority; scalar instruction timing and native WGMMA scheduling are not modeled.",
            "block_DK maps the gated-state reduction; the causal-local GEMM reduces the complete chunk.",
        ],
    )
