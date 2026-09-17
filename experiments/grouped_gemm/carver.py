"""Carver adapter for the padded grouped-GEMM CTA domain."""


def carver_rank(workload, device, configs, top_k):
    from experiments.gemm.carver import rank_configs
    from tilelang.carver.template import GroupedMatmulTemplate

    p = workload.parameters
    block_m = configs[0]["block_M"]
    template = GroupedMatmulTemplate(
        batch_sizes=list(p["batch_sizes"]),
        block_m=block_m,
        N=p["n"],
        K=p["k"],
        trans_B=p.get("transpose_b", False),
        in_dtype=workload.dtype,
        out_dtype=workload.dtype,
        accum_dtype="float32",
    )
    converted = [dict(c, thread_num=c["threads"]) for c in configs]
    for config in converted:
        config.pop("threads")
    report = rank_configs(
        converted,
        m=template.M,
        n=p["n"],
        k=p["k"],
        dtype=workload.dtype,
        target=device.target,
        top_k=top_k,
        transpose_b=p.get("transpose_b", False),
        template=template,
    )
    for record, config in zip(report["configs"], configs):
        record["config"] = config
    report.update(template="GroupedMatmulTemplate", metric="carver_traffic_waves", score_units="byte-waves")
    return report
