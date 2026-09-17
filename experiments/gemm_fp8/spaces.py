"""TCGen05 FP8 GEMM tile, pipeline, thread, and rasterization candidates."""

from experiments.utils.grid import grid


def get_configs():
    return grid(
        block_M=[32, 64, 96, 128, 192, 256],
        block_N=[32, 64, 96, 128, 192, 256],
        block_K=[32, 64, 96, 128],
        num_stages=[0, 1, 2, 3],
        threads=[128, 256],
        enable_rasteration=[True, False],
    )


def support_reason(workload):
    p = workload.parameters
    if workload.dtype not in ("float8_e4m3fn", "float8_e5m2"):
        return "the FP8 GEMM experiment requires E4M3 or E5M2 inputs"
    if p.get("batch", 1) != 1 or p.get("transpose_a", False) or not p.get("transpose_b", False) or p.get("epilogue", "none") != "none":
        return "the FP8 GEMM example requires nonbatched A=(M,K), B=(N,K), transpose_b=True and no fused epilogue"
    return None


def legality_reason(w, device, c):
    return None


def canonical_config(w, c, device=None):
    return dict(c)
