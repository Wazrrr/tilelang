"""H200 candidates expanding the original GEMM grid with finer N tiles."""

from experiments.utils.grid import grid


def candidate_configs():
    return grid(
        block_M=[64, 128, 256],
        block_N=[32, 64, 96, 128, 192, 256],
        block_K=[32, 64],
        num_stages=[0, 1, 2, 3],
        thread_num=[128, 256],
        enable_rasteration=[True, False],
    )


def get_configs():
    from experiments.utils.compiled_pool import compiled_configs

    return compiled_configs("gemm", candidate_configs())


def support_reason(workload):
    p = workload.parameters
    if workload.dtype not in ("float16", "bfloat16"):
        return "the GEMM example experiment supports float16 and bfloat16"
    if p.get("batch", 1) != 1 or p.get("transpose_a", False) or not p.get("transpose_b", False) or p.get("epilogue", "none") != "none":
        return "the GEMM example requires nonbatched A=(M,K), B=(N,K), transpose_b=True and no fused epilogue"
    return None


def legality_reason(w, device, c):
    # The active pool is qualified before ranking; no model-time pruning.
    return None


def canonical_config(w, c, device=None):
    return dict(c)
