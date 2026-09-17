"""One 192-config pool; fixed M tiles keep group metadata identical across it."""

from experiments.utils.grid import grid

BLOCK_M = 64


def get_configs():
    return grid(
        block_M=[BLOCK_M],
        block_N=[32, 64, 96, 128, 192, 256],
        block_K=[16, 32, 48, 64],
        num_stages=[0, 1, 2, 3],
        threads=[128, 256],
    )


def support_reason(workload):
    if workload.dtype not in ("float16", "bfloat16"):
        return "the grouped GEMM example experiment supports float16 and bfloat16"
    return None


def legality_reason(workload, device, config):
    return None


def canonical_config(workload, config, device=None):
    return dict(config)
