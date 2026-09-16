"""One 720-config pool using the example's native scheduling parameters."""

from experiments.utils.grid import grid


def get_configs():
    return grid(
        block_DK=[16, 32, 48, 64, 96, 128],
        block_DV=[16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256],
        num_stages=[0, 1, 2, 3, 4],
        threads=[128, 256],
    )


def legality_reason(workload, device, config):
    # Attempt every declared candidate; retain actual compiler/check failures.
    return None


def canonical_config(workload, config, device=None):
    return dict(config)
