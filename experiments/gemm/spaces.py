"""B200-compilable TCGen05 GEMM candidates."""

from experiments.utils.grid import grid


_EXAMPLE_MAIN_CONFIG = dict(
    block_M=128,
    block_N=128,
    block_K=128,
    num_stages=2,
    thread_num=256,
    enable_rasteration=False,
)


def _compiles_on_b200(config):
    """Encode the exact TCGen05 lowering domain from the B200 census."""
    block_m, block_n, threads = (config[key] for key in ("block_M", "block_N", "thread_num"))
    if block_m in (128, 256):
        return True
    if block_n not in (64, 128, 192, 256):
        return False
    return threads == 128 or (block_m in (32, 64) and block_n != 192)


def get_configs():
    # Keep every configuration from the advanced example while trimming the
    # experiment-only expansion to roughly the size of the other family pools.
    configs = [
        config
        for config in grid(
            block_M=[32, 64, 96, 128, 192, 256],
            block_N=[64, 128, 192, 256],
            block_K=[32, 64],
            num_stages=[0, 1, 2, 3],
            thread_num=[128, 256],
            enable_rasteration=[True, False],
        )
        if _compiles_on_b200(config)
    ]
    # The active SM100 example has one explicit 128x128x128 launch outside the
    # historical autotuning grid. Keep it as an exact, compiler-verified member.
    configs.append(dict(_EXAMPLE_MAIN_CONFIG))
    return configs


def support_reason(workload):
    p = workload.parameters
    if workload.dtype not in ("float16", "bfloat16"):
        return "the GEMM example experiment supports float16 and bfloat16"
    if p.get("batch", 1) != 1 or p.get("transpose_a", False) or not p.get("transpose_b", False) or p.get("epilogue", "none") != "none":
        return "the GEMM example requires nonbatched A=(M,K), B=(N,K), transpose_b=True and no fused epilogue"
    return None


def legality_reason(w, device, c):
    # The declared pool is the compiler-verified SM100 domain.
    return None


def canonical_config(w, c, device=None):
    return dict(c)
