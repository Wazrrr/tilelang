"""Common-grid baselines with explicit workload support boundaries."""

from importlib import import_module


def carver_target_support_reason(target):
    """Respect the unchanged CUDA model's architecture boundary before launch."""
    if target["kind"] != "cuda":
        return "the existing Carver comparison adapter requires CUDA"
    arch = target.get("arch", "")
    version = arch.removeprefix("sm_").rstrip("af")
    # Original precision dispatch covers Volta, Ampere, Ada and Hopper only.
    # Do not map Blackwell to an older GPU just to obtain a baseline score.
    if not version.isdigit() or not 70 <= int(version) <= 90:
        return f"the unchanged Carver CUDA model does not support {arch}; no substitute architecture is used"
    return None


def carver_support_reason(workload, device):
    from experiments.families import FAMILIES

    family = FAMILIES.get(workload.op)
    if family is None:
        return "no Carver template is registered for this operation"
    return import_module(f"experiments.{family}.carver").carver_support_reason(workload, device)


def carver_rank(workload, device, configs, top_k):
    reason = carver_support_reason(workload, device)
    if reason:
        raise ValueError(reason)
    from experiments.families import FAMILIES

    family = FAMILIES[workload.op]
    return import_module(f"experiments.{family}.carver").carver_rank(workload, device, configs, top_k)


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
