"""Common-grid baselines with explicit workload support boundaries."""


def carver_support_reason(workload, device):
    from experiments.common.spec import support_reason

    if device.target["kind"] != "cuda":
        return "Carver experiment templates currently require CUDA"
    return support_reason(workload, device)


def carver_rank(workload, device, configs, top_k):
    reason = carver_support_reason(workload, device)
    if reason:
        raise ValueError(reason)
    if workload.op == "gemm":
        from experiments.gemm.carver import carver_rank

        return carver_rank(workload, device, configs, top_k)
    if workload.op == "gemm_fp8":
        from experiments.gemm.carver import rank_configs

        native = [{("thread_num" if k == "threads" else k): v for k, v in c.items()} for c in configs]
        report = rank_configs(
            native, **{k: workload.parameters[k] for k in ("m", "n", "k")}, dtype=workload.dtype, target=device.target, top_k=top_k
        )
        for row, config in zip(report["configs"], configs):
            row["config"] = dict(config)
        return dict(report, metric="carver_traffic_waves", score_units="byte-waves", template="MatmulTemplate")
    from .carver_graph import rank_graph

    return rank_graph(workload, device, configs, top_k)


def exhaustive_selection(configs):
    """No analytical gates, scores, or candidate measurements enter selection."""
    return dict(
        metric="exhaustive",
        score_units=None,
        selection=dict(requested_k=len(configs), selected_indices=list(range(len(configs))), selected_count=len(configs), wall_time_ms=0),
        ranking=[dict(index=i, rank=i + 1, score=None) for i in range(len(configs))],
        configs=[dict(index=i, config=c, selected=True, status="selected") for i, c in enumerate(configs)],
    )


def random_selection(workload, configs, top_k, seed):
    """Seeded configuration hashes; no outcomes and no failed-trial replacement."""
    from experiments.xgboost.data import canonical_workload, digest

    order = sorted(range(len(configs)), key=lambda i: digest(dict(seed=seed, workload=canonical_workload(workload), config=configs[i])))
    selected = order[:top_k]
    return dict(
        metric="random_config_hash_v1",
        score_units=None,
        selection=dict(
            requested_k=top_k,
            selected_indices=selected,
            selected_count=len(selected),
            wall_time_ms=0,
            failure_policy="no replacement",
            seed=seed,
        ),
        ranking=[dict(index=i, rank=rank + 1, score=None) for rank, i in enumerate(order)],
        configs=[
            dict(index=i, config=c, selected=i in selected, status="selected" if i in selected else "not_selected")
            for i, c in enumerate(configs)
        ],
    )
