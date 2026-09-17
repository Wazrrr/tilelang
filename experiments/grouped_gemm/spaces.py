"""The two native SM100 grouped-MXFP8 schedules: tiled and persistent."""

BLOCK_M = 128
BLOCK_N = 256
BLOCK_K = 128
NUM_STAGES = 6


def get_configs():
    common = dict(block_M=BLOCK_M, block_N=BLOCK_N, block_K=BLOCK_K, num_stages=NUM_STAGES)
    return [dict(common, threads=128, persistent=False), dict(common, threads=256, persistent=True)]


def support_reason(workload):
    if workload.dtype != "float8_e4m3fn":
        return "the SM100 grouped GEMM experiment requires MXFP8 E4M3 inputs"
    p = workload.parameters
    if p["n"] % BLOCK_N or p["k"] % BLOCK_K:
        return "the SM100 grouped MXFP8 example requires N divisible by 256 and K divisible by 128"
    return None


def legality_reason(workload, device, config):
    # Record actual compilation/correctness failures for every declared candidate.
    return None


def canonical_config(workload, config, device=None):
    return dict(config)
