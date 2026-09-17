"""Common-grid baselines with explicit workload support boundaries."""


def carver_support_reason(workload, device):
    if device.target["kind"] != "cuda":
        return "the Carver experiment adapters require CUDA"
    from .spec import support_reason

    reason = support_reason(workload, device)
    if reason:
        return reason
    from experiments.families import family_module

    try:
        adapter = family_module(workload.op, "carver")
    except ModuleNotFoundError:
        return f"no Carver template adapter for {workload.op}"
    adapter_reason = getattr(adapter, "support_reason", lambda _workload, _device: None)(workload, device)
    if adapter_reason:
        return adapter_reason
    return None


def carver_rank(workload, device, configs, top_k):
    reason = carver_support_reason(workload, device)
    if reason:
        raise ValueError(reason)
    from experiments.families import family_module

    return family_module(workload.op, "carver").carver_rank(workload, device, configs, top_k)


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
