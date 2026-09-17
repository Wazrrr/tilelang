"""Carver adapter for the FP8 GEMM example's exact pool."""


def carver_rank(workload, device, configs, top_k):
    from experiments.gemm.carver import rank_configs

    p = workload.parameters
    converted = [dict(c, thread_num=c["threads"]) for c in configs]
    for config in converted:
        config.pop("threads")
    dtype = "float8_e4m3" if workload.dtype == "float8_e4m3fn" else workload.dtype
    report = rank_configs(
        converted,
        m=p["m"],
        n=p["n"],
        k=p["k"],
        dtype=dtype,
        target=device.target,
        top_k=top_k,
        transpose_a=p.get("transpose_a", False),
        transpose_b=p.get("transpose_b", False),
    )
    for record, config in zip(report["configs"], configs):
        record["config"] = config
    report.update(template="MatmulTemplate", metric="carver_traffic_waves", score_units="byte-waves")
    return report
