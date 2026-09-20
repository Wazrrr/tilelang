"""Compiler-free configuration IDs and audit records for complete pools."""

from collections import Counter
import hashlib
import json

SPACE_VERSION = 8
PRESETS = ("expanded",)


def config_id(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def legality_reason(w, device, c):
    from experiments.families import family_module

    if device.target["kind"] not in ("cuda", "hip"):
        return "expanded native GPU schedules require CUDA or HIP"
    return family_module(w.op, "spaces").legality_reason(w, device, c)


def canonical_config(w, c, device=None):
    from experiments.families import family_module

    return family_module(w.op, "spaces").canonical_config(w, c, device)


def audit_space(workload, configs, *, explicit=False, retained_current_count=0, generated_count=None, rejected=(), aliases=()):
    configs = [dict(c) for c in configs]
    ids = [config_id(c) for c in configs]
    if len(ids) != len(set(ids)):
        raise ValueError("configuration pool requires unique configurations")
    rejected, aliases = list(rejected), list(aliases)
    generated = len(configs) if generated_count is None else generated_count
    return dict(
        version=SPACE_VERSION,
        preset="explicit" if explicit else workload.config_space,
        configs=configs,
        config_ids=ids,
        generated_count=generated,
        candidate_count=len(configs),
        retained_current_count=retained_current_count,
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


def space_summary(space):
    return {k: v for k, v in space.items() if k not in ("configs", "config_ids", "aliases", "rejected")}
