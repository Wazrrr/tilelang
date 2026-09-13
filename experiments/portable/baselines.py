"""Common-grid baselines with explicit workload support boundaries."""


def carver_support_reason(workload, device):
    if device.target["kind"] != "cuda":
        return "the existing Carver comparison adapter requires CUDA"
    if workload.op != "gemm" or workload.dtype not in ("float16", "bfloat16"):
        return "the existing Carver comparison adapter supports FP16/BF16 GEMM only"
    if workload.parameters.get("batch", 1) != 1 or workload.parameters.get("epilogue", "none") != "none":
        return "the existing Carver comparison adapter has no batched or fused-epilogue model"
    return None


def carver_rank(workload, device, configs, top_k):
    from experiments.gemm.carver import rank_configs

    reason = carver_support_reason(workload, device)
    if reason:
        raise ValueError(reason)
    aliases = dict(block_m="block_M", block_n="block_N", block_k="block_K", stages="num_stages", threads="thread_num")
    grid = [{aliases[key]: value for key, value in config.items()} for config in configs]
    p = workload.parameters
    report = rank_configs(
        grid,
        m=p["m"],
        n=p["n"],
        k=p["k"],
        dtype=workload.dtype,
        target=device.target,
        top_k=top_k,
        transpose_a=p.get("transpose_a", False),
        transpose_b=p.get("transpose_b", False),
    )
    for record, config in zip(report["configs"], configs):
        record["config"] = config
    report.update(metric="carver_traffic_waves", score_units="byte-waves")
    return report


def exhaustive_selection(configs):
    """No analytical gates, scores, or candidate measurements enter selection."""
    return dict(
        metric="exhaustive",
        score_units=None,
        selection=dict(requested_k=len(configs), selected_indices=list(range(len(configs))), selected_count=len(configs), wall_time_ms=0),
        ranking=[dict(index=i, rank=i + 1, score=None) for i in range(len(configs))],
        configs=[dict(index=i, config=c, selected=True, status="selected") for i, c in enumerate(configs)],
    )
