"""SM100 MHA-forward tiles, pipeline depths, and SS/TS variants."""

from experiments.utils.grid import grid


def get_configs():
    return grid(
        # The single-CTA SM100 attention kernel uses one TMEM datapath row per
        # query row. Larger M tiles can compile but deadlock for longer causal
        # sequences, so they are not meaningful members of the common pool.
        block_M=[32, 64, 128],
        block_N=list(range(16, 257, 16)),
        num_stages=[0, 1, 2, 3, 4, 5],
        threads=[128, 256],
    )


def legality_reason(workload, device, config):
    # Attempt every declared candidate; retain actual compiler/check failures.
    return None


def canonical_config(workload, config, device=None):
    return dict(config)
