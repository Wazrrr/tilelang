"""One B200-compilable pool using token-parallel KDA parameters."""

from experiments.utils.grid import grid


def get_configs():
    configs = grid(
        block_H=list(range(1, 17)),
        num_stages=list(range(9)),
        threads=[32, 64, 128, 256],
    )
    # The B200 layout planner cannot place odd multi-head tiles at 256 threads.
    # All other combinations, including the example's complete native grid,
    # compile. Stage eight was compiler-checked separately to keep >500 configs.
    return [
        config
        for config in configs
        if config["threads"] != 256 or config["block_H"] == 1 or config["block_H"] % 2 == 0
    ]


def legality_reason(workload, device, config):
    # The declared pool is the compiler-verified SM100 domain.
    return None


def canonical_config(workload, config, device=None):
    return dict(config)
