"""Carver adapter for the FP8 GEMM example's exact pool."""


def carver_rank(workload, device, configs, top_k):
    from experiments.common.carver import _architecture, workload_template
    from experiments.gemm.carver import rank_configs

    p = workload.parameters
    template = workload_template(workload, configs, arch=_architecture(device.target))
    report = rank_configs(
        configs,
        m=p["m"],
        n=p["n"],
        k=p["k"],
        dtype=template.in_dtype,
        target=device.target,
        top_k=top_k,
        transpose_a=p.get("transpose_a", False),
        transpose_b=p.get("transpose_b", False),
        template=template,
        thread_key="threads",
    )
    report.update(metric="carver_traffic_waves", score_units="byte-waves")
    return report
