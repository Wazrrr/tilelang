"""One 512-config pool using token-parallel KDA scheduling parameters."""

from experiments.utils.grid import grid


def get_configs():
    return grid(
        block_H=list(range(1, 17)),
        num_stages=list(range(8)),
        threads=[32, 64, 128, 256],
    )


def legality_reason(workload, device, config):
    # Attempt every declared candidate; retain actual compiler/check failures.
    return None


def canonical_config(workload, config, device=None):
    return dict(config)
