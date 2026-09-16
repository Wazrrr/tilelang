"""One 320-config pool using the example's native scheduling parameters."""

from experiments.utils.grid import grid


def get_configs():
    return grid(
        block_M=[32, 64, 128, 192, 256],
        block_N=[16, 32, 48, 64, 96, 128, 192, 256],
        num_stages=[0, 1, 2, 3],
        threads=[128, 256],
    )


def legality_reason(workload, device, config):
    # Attempt every declared candidate; retain actual compiler/check failures.
    return None


def canonical_config(workload, config, device=None):
    return dict(config)
