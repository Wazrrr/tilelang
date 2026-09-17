"""Carver adapter for the padded grouped-GEMM CTA domain."""


def support_reason(workload, device):
    return "Carver has no block-scaled 2-CTA grouped-MXFP8 model"


def carver_rank(workload, device, configs, top_k):
    reason = support_reason(workload, device)
    if reason:
        raise ValueError(reason)
    from experiments.common.carver import _architecture, workload_template
    from experiments.gemm.carver import rank_configs

    p = workload.parameters
    template = workload_template(workload, configs, arch=_architecture(device.target))
    report = rank_configs(
        configs,
        m=template.M,
        n=p["n"],
        k=p["k"],
        dtype=workload.dtype,
        target=device.target,
        top_k=top_k,
        transpose_b=p.get("transpose_b", False),
        template=template,
        thread_key="threads",
    )
    report.update(metric="carver_traffic_waves", score_units="byte-waves")
    return report
