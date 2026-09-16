"""One 224-config pool using the example's native scheduling parameters."""

from experiments.utils.grid import grid


def get_configs():
    return grid(
        BLOCK_M=[1, 2, 4, 8, 16, 32, 64],
        BLOCK_N=[128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        threads=[64, 128, 256, 512],
    )


def legality_reason(workload, device, config):
    # Attempt every declared candidate; retain actual compiler/check failures.
    return None


def canonical_config(workload, config, device=None):
    return dict(config)
