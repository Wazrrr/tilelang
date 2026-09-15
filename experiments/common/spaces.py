"""Declared schedule domains; planning needs only the Python standard library.

The common pool is independent of TileTune predictions and measured timings.
Current entries keep their indices; newly generated aliases point to them.
"""

from collections import Counter
import hashlib
from itertools import product
import json

from .mma import mma_partition as mma_partition

SPACE_VERSION = 1
PRESETS = ("current", "expanded", "large")
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
        # Generate expanded first so large always retains its indices.
        for large in [False, True] if w.config_space == "large" else [False]:
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
    return dict(
        version=SPACE_VERSION,
        preset="explicit" if explicit else w.config_space,
        configs=configs,
        config_ids=ids,
        generated_count=generated,
        candidate_count=len(configs),
        retained_current_count=len(current),
        rejected_count=len(rejected),
        rejection_reasons=dict(Counter(r["reason"] for r in rejected)),
        alias_count=len(aliases),
        aliases=aliases,
        rejected=rejected,
        compiled_count=None,
        distinct_program_count=None,
        correct_count=None,
    )


def space_summary(space):
    return {k: v for k, v in space.items() if k not in ("configs", "config_ids", "aliases", "rejected")}
