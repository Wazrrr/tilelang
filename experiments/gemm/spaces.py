"""One 576-config expansion of the advanced example's 288-config grid."""

from experiments.utils.grid import grid


def get_configs():
    return grid(
        block_M=[64, 128, 256],
        block_N=[64, 128, 256],
        block_K=[16, 32, 48, 64],
        num_stages=[0, 1, 2, 3],
        thread_num=[128, 256],
        enable_rasteration=[True, False],
    )


def support_reason(workload):
    p = workload.parameters
    if workload.dtype not in ("float16", "bfloat16"):
        return "the GEMM example experiment supports float16 and bfloat16"
    if p.get("batch", 1) != 1 or p.get("transpose_a", False) or not p.get("transpose_b", False) or p.get("epilogue", "none") != "none":
        return "the GEMM example requires nonbatched A=(M,K), B=(N,K), transpose_b=True and no fused epilogue"
    return None


def legality_reason(w, device, c):
    # Compile every declared candidate; record actual compiler/correctness failures.
    return None


def canonical_config(w, c, device=None):
    return dict(c)
