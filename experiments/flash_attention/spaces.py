"""H200 attention candidates, including larger query tiles."""

from experiments.utils.grid import grid


def candidate_configs():
    return grid(
        block_M=[32, 64, 128, 256],
        block_N=list(range(16, 257, 16)),
        num_stages=[0, 1, 2, 3, 4, 5, 6, 7],
        threads=[128, 256],
    )


def get_configs():
    from experiments.utils.compiled_pool import compiled_configs

    return compiled_configs("flash_attention", candidate_configs())


def legality_reason(workload, device, config):
    # The active pool is qualified before ranking; no model-time pruning.
    return None


def canonical_config(workload, config, device=None):
    return dict(config)
