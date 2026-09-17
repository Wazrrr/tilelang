"""Ampere FP8-storage/BF16-compute schedules with one fixed scale layout."""

from experiments.utils.grid import grid

BLOCK_K = 128


def get_configs():
    return grid(
        block_M=[32, 64, 128],
        block_N=[32, 64, 128],
        block_K=[BLOCK_K],
        num_stages=[0, 1, 2, 3],
        threads=[128, 256],
    )


def support_reason(workload, device=None):
    if workload.dtype != "float8_e4m3fn" or not workload.parameters.get("transpose_b", False):
        return "the Ampere block-scaled experiment requires E4M3 A=(M,K), B=(N,K), transpose_b=True"
    p = workload.parameters
    if p["m"] % 32 or p["n"] % 32 or p["k"] % BLOCK_K:
        return "the fixed Ampere scale layout requires M%32=0, N%32=0, and K%128=0"
    if device is not None and device.target["kind"] != "cuda":
        return "the Ampere FP8-storage emulation requires CUDA"
    return None


def legality_reason(workload, device, config):
    return None


def canonical_config(workload, config, device=None):
    return dict(config)
