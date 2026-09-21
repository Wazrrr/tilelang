"""Candidate KDA schedules; compilation qualification defines the final pool."""

from experiments.utils.grid import grid


def candidate_configs():
    return grid(
        block_H=list(range(1, 17)),
        num_stages=list(range(16)),
        threads=[32, 64, 128, 256],
    )


def get_configs():
    from experiments.utils.compiled_pool import compiled_configs

    return compiled_configs("kda", candidate_configs())


def legality_reason(workload, device, config):
    # The active pool is qualified before ranking; no model-time pruning.
    return None


def canonical_config(workload, config, device=None):
    return dict(config)
