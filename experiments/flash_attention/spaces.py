"""One 480-config pool of FullRow-compatible attention tiles."""

from experiments.utils.grid import grid

# The example's native autotune config (64/64/1/128) and its explicit
# 128/128/1/128 launch are always retained in the expanded pool.
_EXAMPLE_CONFIGS = (
    dict(block_M=64, block_N=64, num_stages=1, threads=128),
    dict(block_M=128, block_N=128, num_stages=1, threads=128),
)


def _valid_thread_tile(block_M, threads):
    # T.gemm with policy=FullRow requires each warp to own a multiple of 16
    # rows. With threads / 32 warps, block_M * 32 / threads must be a multiple
    # of 16, which simplifies to block_M * 2 % threads == 0.
    return (block_M * 2) % threads == 0


def get_configs():
    configs = [
        config
        for config in grid(
            # Slice the query tile one step slimmer than the original
            # 32/64/128 sweep and allow the matching single-warp layout.
            block_M=[16, 32, 64, 128],
            block_N=list(range(16, 257, 16)),
            # num_stages is capped at 2: on Ampere, 3+ stages push the K/V
            # pipeline buffers past the 164 KB shared-memory limit and fail
            # at kernel launch, so they are not part of the declared pool.
            num_stages=[0, 1, 2],
            threads=[32, 64, 128, 256],
        )
        if _valid_thread_tile(config["block_M"], config["threads"])
    ]
    for config in _EXAMPLE_CONFIGS:
        if config not in configs:
            configs.append(config)
    return configs


def legality_reason(workload, device, config):
    if not _valid_thread_tile(config["block_M"], config["threads"]):
        return (
            "T.gemm FullRow requires block_M * 2 to be divisible by threads "
            "so every warp owns a multiple of 16 rows"
        )
    return None


def canonical_config(workload, config, device=None):
    return dict(config)
