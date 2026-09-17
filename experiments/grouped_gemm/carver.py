"""Carver's grouped template, with the original per-graph policy score."""


def carver_support_reason(workload, device):
    from experiments.common.baselines import carver_target_support_reason
    from .spaces import support_reason

    return carver_target_support_reason(device.target) or support_reason(workload)


def carver_rank(workload, device, configs, top_k):
    from tilelang.carver.arch import CUDA
    from tilelang.carver.template import GroupedMatmulTemplate
    from tilelang.carver.matmul_analysis import get_tensorized_func_and_tags
    from tilelang.carver.roller.policy import TensorCorePolicy
    from experiments.gemm.carver import model_target
    from experiments.utils.carver import evaluate_tile, template_report

    reason = carver_support_reason(workload, device)
    if reason:
        raise ValueError(reason)
    arch = CUDA(model_target(device.target))
    p = workload.parameters
    template = GroupedMatmulTemplate(
        batch_sizes=tuple(p["batch_sizes"]),
        N=p["n"],
        K=p["k"],
        trans_B=p.get("transpose_b", False),
        in_dtype=workload.dtype,
        out_dtype=workload.dtype,
        _arch=arch,
    )
    policies = []
    for func in template.equivalent_function():
        func, tags = get_tensorized_func_and_tags(func, arch.target, allow_gemv=True)
        if not tags:
            raise ValueError("Carver cannot tensorize a grouped GEMM member")
        policies.append(TensorCorePolicy.from_prim_func(func, arch, tags))
    evaluations = []
    for config in configs:
        models = []
        for policy in policies:
            policy.pipeline_stage = max(1, config["num_stages"])
            steps = {node: {axis.var.name: config["block_K"] for axis in node.raxis} for node in policy.ordered_nodes}
            models.append(evaluate_tile(policy, [config["block_M"], config["block_N"]], steps, config["threads"]))
        evaluations.append(models)
    return template_report(
        configs,
        evaluations,
        top_k,
        template="carver_grouped_matmul",
        assumptions=[
            "Reuses MatmulTemplate for each group with its exact M, N, K, dtype and B transpose.",
            "Sums unchanged Carver graph scores; group metadata lookup and interleaving of group CTAs are not modeled.",
            "Pipeline stages 0 and 1 use one shared-memory copy; failed policy candidates receive no fallback.",
        ],
    )
