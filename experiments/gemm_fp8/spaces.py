"""Native Hopper schedules for one fixed E4M3 block-scale layout."""

BLOCK_M = 64
BLOCK_K = 128
NUM_STAGES = 4
THREADS = 128


def get_configs():
    return [
        dict(block_M=BLOCK_M, block_N=block_n, block_K=BLOCK_K, num_stages=NUM_STAGES, threads=THREADS)
        for block_n in (16, 32, 64, 128)
    ]


def support_reason(workload, device=None):
    if workload.dtype != "float8_e4m3fn":
        return "block-scaled FP8 GEMM requires E4M3 inputs"
    p = workload.parameters
    if not p.get("transpose_b", False):
        return "block-scaled FP8 GEMM requires A=(M,K), B=(N,K), transpose_b=True"
    if p["m"] % BLOCK_M or p["n"] % 128 or p["k"] % BLOCK_K:
        return "the fixed Hopper scale layout requires M%64=0, N%128=0, and K%128=0"
    if device is not None and device.target["kind"] == "cuda":
        import re

        match = re.fullmatch(r"sm_(\d+)[af]?", device.target["arch"])
        if not match or int(match[1]) < 89:
            return "native FP8 GEMM requires CUDA sm_89 or newer"
    return None


def legality_reason(workload, device, config):
    return None


def canonical_config(workload, config, device=None):
    return dict(config)
