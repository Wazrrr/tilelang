"""B200-compilable SM100 MHA-forward SS/TS configurations."""

from experiments.utils.grid import grid


def get_configs():
    configs = grid(
        block_M=[32, 64, 128],
        block_N=list(range(16, 257, 16)),
        num_stages=list(range(13)),
        threads=[128, 256],
    )
    # TCGen05 accepts every N/variant combination at M=128. The smaller M
    # tiles compile only through the SS path and native 64-column instruction
    # shapes. These rules reproduce the exhaustive B200 compile census.
    return [
        config
        for config in configs
        if config["block_M"] == 128 or (config["threads"] == 128 and config["block_N"] % 64 == 0)
    ]


def legality_reason(workload, device, config):
    # The declared pool is the compiler-verified SM100 domain.
    return None


def canonical_config(workload, config, device=None):
    return dict(config)
