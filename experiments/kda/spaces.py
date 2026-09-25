"""Token-parallel KDA schedule pool that is safe to benchmark on Ampere.

Only the non-pipelined schedule is kept because deeper pipelines
(``num_stages >= 1``) can hang at benchmark time and wedge the CUDA context,
and the block_H/threads combinations that hang or fail CUDA compilation are
removed. The measured oracle winners (``block_H=2`` for the long, medium,
regular and batched cases, ``block_H=8`` for the short case) are retained.
"""

from experiments.utils.grid import grid

# block_H -> thread counts that are safe to actually benchmark on Ampere for the
# token-parallel intra-chunk kernel. Combinations that hang at benchmark time
# (and wedge the CUDA context) are removed. The hang set was measured across all
# five KDA workloads on A100: block_H=3/threads=1024, all of block_H=5, all of
# block_H=6, and block_H=7/threads=32. The measured oracle winners are retained.
_BENCHMARK_SAFE_THREADS = {
    1: (32, 64, 128, 256, 512, 1024),
    2: (32, 64, 128, 256, 512, 1024),
    3: (32, 64, 128, 512),
    4: (32, 64, 128, 256),
    5: (),
    6: (),
    7: (64, 128, 1024),
    8: (32, 64, 128, 256, 512, 1024),
    9: (),
    10: (32, 128),
    11: (),
    12: (32, 64, 128, 256, 512),
    13: (64,),
    14: (32, 64, 128, 256),
    15: (32, 64, 128),
    16: (32, 64, 128, 256, 512, 1024),
}


# Slice the head dimension 128 into ``block_DK`` tiles. The slimmer 4/8/16
# slices are only safe for power-of-two ``block_H`` tiles; for every other
# ``block_H`` layout the token-parallel fragment layout hangs at benchmark time
# (measured: block_H=3 with block_DK=4/8), which wedges the CUDA context. Those
# layouts therefore keep the 32/64/128 slices only.
_POWER_OF_TWO_BLOCK_H = (1, 2, 4, 8, 16)
_SLIMMER_BLOCK_DK = (4, 8, 16)


def get_configs():
    """Return the declared benchmark-safe KDA pool.

    ``block_DK`` is every power-of-two divisor of the head dimension 128 for
    power-of-two ``block_H`` tiles, and the 32/64/128 slices otherwise. This
    yields 234 attempt-safe configs with no hangs.
    """
    return [
        config
        for config in grid(
            block_H=list(range(1, 17)),
            num_stages=[0],
            threads=[32, 64, 128, 256, 512, 1024],
            block_DK=[4, 8, 16, 32, 64, 128],
        )
        if config["threads"] in _BENCHMARK_SAFE_THREADS.get(config["block_H"], ())
        and not (
            config["block_H"] not in _POWER_OF_TWO_BLOCK_H
            and config["block_DK"] in _SLIMMER_BLOCK_DK
        )
    ]


def legality_reason(workload, device, config):
    threads = _BENCHMARK_SAFE_THREADS.get(config["block_H"], ())
    if config["threads"] not in threads:
        return "no valid layout for this block_H/threads combination"
    return None


def canonical_config(workload, config, device=None):
    return dict(config)
