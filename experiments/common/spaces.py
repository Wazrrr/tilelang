"""Declared schedule domains; planning needs only the Python standard library.

Enumeration never reads TileTune predictions or measured timings.
Current entries keep their indices; newly generated aliases point to them.
"""

from collections import Counter
import hashlib
from itertools import product
import json

from .mma import mma_partition as mma_partition

SPACE_VERSION = 2
PRESETS = ("current", "expanded", "large", "exhaustive")
LARGE_CONFIG_LIMIT = 1024
POLICIES = ("square", "full_row", "full_col")


def config_id(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _grid(**axes):
    for values in product(*axes.values()):
        yield dict(zip(axes, values))


def _new_configs(w, large):
    from experiments.families import family_module

    yield from family_module(w.op, "spaces").expanded_configurations(w, large)


def legality_reason(w, device, c):
    """Ask the family for structural constraints, independently of ranking."""
    from experiments.families import family_module

    if device.target["kind"] not in ("cuda", "hip"):
        return "expanded native GPU schedules require CUDA or HIP"
    return family_module(w.op, "spaces").legality_reason(w, device, c)


def canonical_config(w, c, device=None):
    """Delegate implementation defaults and equivalence rules to the family."""
    from experiments.families import family_module

    return family_module(w.op, "spaces").canonical_config(w, c, device)


def expand_space(w, device, current, *, explicit=False):
    configs = [dict(c) for c in current]
    ids = [config_id(c) for c in configs]
    if len(ids) != len(set(ids)):
        raise ValueError("configuration pool requires unique configurations")
    rejected, aliases, generated = [], [], len(configs)
    if not explicit and w.config_space != "current":
        seen = {}
        for i, c in enumerate(configs):
            seen.setdefault(config_id(canonical_config(w, c, device)), i)
        # Exhaustive preserves the old large pool and its original indices.
        for large in [False, True] if w.config_space in ("large", "exhaustive") else [False]:
            for c in _new_configs(w, large):
                generated += 1
                reason = legality_reason(w, device, c)
                if reason:
                    rejected.append(dict(config=c, reason=reason))
                    continue
                key = config_id(canonical_config(w, c, device))
                if key in seen:
                    aliases.append(dict(config=c, index=seen[key]))
                    continue
                seen[key] = len(configs)
                configs.append(c)
                ids.append(config_id(c))
    space = dict(
        version=SPACE_VERSION,
        preset="explicit" if explicit else w.config_space,
        configs=configs,
        config_ids=ids,
        generated_count=generated,
        candidate_count=len(configs),
        retained_current_count=len(current),
        budget_omitted_count=0,
        rejected_count=len(rejected),
        rejection_reasons=dict(Counter(r["reason"] for r in rejected)),
        alias_count=len(aliases),
        aliases=aliases,
        rejected=rejected,
        compiled_count=None,
        distinct_program_count=None,
        correct_count=None,
    )
    if not explicit and w.config_space == "large" and len(configs) > LARGE_CONFIG_LIMIT:
        _limit_large_space(w, device, space)
    return space


def _limit_large_space(workload, device, space):
    from experiments.families import family_module
    from .subsets import pairwise_subset

    configs = space["configs"]
    protected = getattr(family_module(workload.op, "spaces"), "protected_configurations", lambda w: ())
    keys = {config_id(canonical_config(workload, c, device)) for c in protected(workload)}
    required = [
        i for i, c in enumerate(configs) if i < space["retained_current_count"] or config_id(canonical_config(workload, c, device)) in keys
    ]
    selection = pairwise_subset(workload, device, configs, LARGE_CONFIG_LIMIT, required_indices=required)
    indices = selection["indices"]
    remap = {old: new for new, old in enumerate(indices)}
    # An alias of an omitted candidate has no index in the compact pool.
    # Keep its exhaustive index so every generated entry remains auditable.
    for alias in space["aliases"]:
        alias["exhaustive_index"] = alias["index"]
        alias["index"] = remap.get(alias["index"])
    space.update(
        configs=[configs[i] for i in indices],
        config_ids=selection["config_ids"],
        candidate_count=len(indices),
        budget_omitted_count=len(configs) - len(indices),
        selection=dict(selection, policy="protected_then_greedy_pairwise_then_canonical_hash", source_preset="exhaustive"),
    )


def space_summary(space):
    summary = {k: v for k, v in space.items() if k not in ("configs", "config_ids", "aliases", "rejected")}
    if "selection" in summary:
        summary["selection"] = {
            k: v for k, v in summary["selection"].items() if k not in ("indices", "config_ids", "required_indices", "aliases", "rejected")
        }
        summary["selection"]["required_count"] = len(space["selection"]["required_indices"])
    return summary
