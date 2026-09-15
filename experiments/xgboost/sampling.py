"""Reproducible configuration sampling without inspecting candidate outcomes."""

from collections import defaultdict
import math

from .data import digest, domain, workload_key

DEFAULT_SAMPLE_FRACTION = 0.1
SAMPLING_POLICY = "seeded_config_hash_v1"
STRATIFIED_POLICY = "implementation_stratified_config_hash_v1"
SAMPLING_POLICIES = (SAMPLING_POLICY, STRATIFIED_POLICY)


def sampling_plan(workload, configs, *, fraction=DEFAULT_SAMPLE_FRACTION, seed=123, policy=SAMPLING_POLICY):
    """Freeze a subset of the supplied pool using only workload/config inputs."""
    if isinstance(fraction, bool) or not isinstance(fraction, int | float) or not 0 < fraction <= 1:
        raise ValueError("sample fraction must be in (0, 1]")
    if type(seed) is not int or seed < 0:
        raise ValueError("sampling seed must be a nonnegative integer")
    if policy not in SAMPLING_POLICIES:
        raise ValueError("unsupported sampling policy")
    if not configs or len({digest(config) for config in configs}) != len(configs):
        raise ValueError("sampling requires a nonempty pool of unique configurations")
    count = max(1, math.ceil(len(configs) * fraction))
    order = sorted(range(len(configs)), key=lambda i: digest(dict(seed=seed, workload=workload, config=configs[i])))
    selected, coverage = order[:count], None
    if policy == STRATIFIED_POLICY:
        strata = defaultdict(list)
        for i in order:
            strata[str(configs[i].get("implementation", "original"))].append(i)
        names = sorted(strata)
        quotas = {name: 0 for name in names}
        remaining = count
        # Up to three attempted examples per implementation, including when a
        # small budget cannot give every stratum its full initial quota.
        for _ in range(3):
            for name in names:
                if remaining and quotas[name] < len(strata[name]):
                    quotas[name] += 1
                    remaining -= 1
        capacities = {name: len(strata[name]) - quotas[name] for name in names}
        capacity = sum(capacities.values())
        if remaining:
            # Integer largest-remainder allocation avoids floating point ties.
            extras = {name: remaining * capacities[name] // capacity for name in names}
            remainder = remaining - sum(extras.values())
            priority = sorted(names, key=lambda name: (-(remaining * capacities[name] % capacity), name))
            for name in priority[:remainder]:
                extras[name] += 1
            quotas = {name: quotas[name] + extras[name] for name in names}
        selected = [i for name in names for i in strata[name][: quotas[name]]]
        coverage = [
            dict(
                implementation=name,
                capacity=len(strata[name]),
                attempted_quota=quotas[name],
                unmet_initial_quota=max(0, min(3, len(strata[name])) - quotas[name]),
            )
            for name in names
        ]
    result = dict(
        policy=policy,
        fraction=fraction,
        seed=seed,
        pool_configs=configs,
        selected_indices=sorted(selected),
    )
    if coverage is not None:
        result["strata"] = coverage
    return result


def sample_runs(runs, *, fraction=DEFAULT_SAMPLE_FRACTION, seed=123, policy=SAMPLING_POLICY):
    """Sample source pools once, including when collection was already sampled.

    Repeated runs share one selection over their union of configurations.
    Failed selected candidates are never replaced with successful alternatives.
    Collection durations retain their original scope for retrospective sampling.
    """
    groups = defaultdict(list)
    for run in runs:
        groups[digest(run["context"])].append(run)
    sampled, report = [], []
    for key in sorted(groups):
        sources = groups[key]
        context = sources[0]["context"]
        pool = {}
        for run in sources:
            collection = run.get("sampling")
            if collection is not None:
                expected = sampling_plan(context["workload"], collection["pool_configs"], fraction=fraction, seed=seed, policy=policy)
                if collection != expected or run["configs"] != [expected["pool_configs"][i] for i in expected["selected_indices"]]:
                    raise ValueError("collected XGBoost sample differs from the requested sampling plan")
            configs = collection["pool_configs"] if collection is not None else run["configs"]
            pool.update((digest(config), config) for config in configs)
        configs = [pool[key] for key in sorted(pool)]
        plan = sampling_plan(context["workload"], configs, fraction=fraction, seed=seed, policy=policy)
        selected = [configs[i] for i in plan["selected_indices"]]
        selected_keys = {digest(config) for config in selected}
        measured = set()
        for run in sources:
            samples = [sample for sample in run["samples"] if digest(sample["config"]) in selected_keys]
            measured.update(digest(sample["config"]) for sample in samples)
            sampled.append({**run, "samples": samples})
        coverage = []
        for implementation in sorted({str(c.get("implementation", "original")) for c in configs}):
            members = [c for c in configs if str(c.get("implementation", "original")) == implementation]
            coverage.append(
                dict(
                    implementation=implementation,
                    pool_size=len(members),
                    selected_count=sum(digest(c) in selected_keys for c in members),
                    successful_count=sum(digest(c) in measured for c in members),
                )
            )
        report.append(
            dict(
                workload=workload_key(context),
                execution_domain=digest(domain(context)),
                pool_size=len(configs),
                selected_configs=selected,
                selected_count=len(selected),
                measured_count=len(measured),
                collection_sampled=all(run.get("sampling") is not None for run in sources),
                strata=plan.get("strata"),
                implementation_coverage=coverage,
            )
        )
    return sampled, report
