"""Native SM100 block-scaled E4M3 schedules using TCGen05."""

BLOCK_M = 128
BLOCK_K = 128
NUM_STAGES = (2, 3, 4, 5, 6)
GROUP_SIZES = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16)
STORE_BLOCK_NS = (16, 32, 64, 128)


def get_configs():
    common = dict(block_M=BLOCK_M, block_N=256, block_K=BLOCK_K)
    configs = [
        dict(
            common,
            num_stages=num_stages,
            threads=128,
            implementation="tcgen05_2cta",
            group_size=1,
            column_major=True,
            use_tma_store=True,
            store_block_N=64,
        )
        for num_stages in NUM_STAGES
    ]
    configs += [
        dict(
            common,
            num_stages=num_stages,
            threads=256,
            implementation="tcgen05_2cta_persistent",
            group_size=group_size,
            column_major=column_major,
            use_tma_store=True,
            store_block_N=store_block_n,
        )
        for num_stages in NUM_STAGES
        for group_size in GROUP_SIZES
        for column_major in (True, False)
        for store_block_n in STORE_BLOCK_NS
        if not (num_stages == 6 and store_block_n == 128)
    ]
    configs += [
        dict(
            common,
            num_stages=num_stages,
            threads=256,
            implementation="tcgen05_2cta_persistent",
            group_size=group_size,
            column_major=column_major,
            use_tma_store=False,
            store_block_N=16,
        )
        for num_stages in NUM_STAGES
        for group_size in GROUP_SIZES
        for column_major in (True, False)
    ]
    return configs


def support_reason(workload, device=None):
    if workload.dtype != "float8_e4m3fn":
        return "SM100 block-scaled FP8 GEMM requires E4M3 inputs"
    p = workload.parameters
    if not p.get("transpose_b", False):
        return "SM100 block-scaled FP8 GEMM requires A=(M,K), B=(N,K), transpose_b=True"
    if p["m"] % (2 * BLOCK_M) or p["n"] % 256 or p["k"] % (2 * BLOCK_K):
        return "the two-CTA SM100 kernel requires M%256=0, N%256=0, and K%256=0"
    if device is not None:
        import re

        if device.target["kind"] != "cuda":
            return "SM100 block-scaled FP8 GEMM requires CUDA"
        match = re.fullmatch(r"sm_(\d+)[af]?", device.target["arch"])
        if not match or int(match[1]) < 100:
            return "TCGen05 block-scaled FP8 GEMM requires CUDA sm_100 or newer"
    return None


def legality_reason(workload, device, config):
    return None


def canonical_config(workload, config, device=None):
    return dict(config)
