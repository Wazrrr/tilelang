"""H200 FP8 schedules containing the original example's complete grid."""

from experiments.utils.grid import grid


def candidate_configs():
    return grid(
        block_M=[64, 128, 256],
        block_N=[64, 128, 256],
        block_K=[32, 64],
        num_stages=list(range(8)),
        threads=[128, 256],
        enable_rasteration=[True, False],
    )


def get_configs():
    from experiments.utils.compiled_pool import compiled_configs

    return compiled_configs("gemm_fp8", candidate_configs())


def support_reason(workload, device=None):
    p = workload.parameters
    if workload.dtype != "float8_e4m3fn" or not p.get("transpose_b", False):
        return "H200 FP8 GEMM requires E4M3 A=(M,K), B=(N,K), transpose_b=True"
    if p.get("batch", 1) != 1 or p.get("transpose_a", False) or p.get("epilogue", "none") != "none":
        return "H200 FP8 GEMM requires nonbatched operands and no fused epilogue"
    if device is not None:
        arch = device.target.get("arch", "").removeprefix("sm_").rstrip("af")
        minimum = 89
        if device.target["kind"] != "cuda" or not arch.isdigit() or int(arch) < minimum:
            return f"this FP8 implementation requires CUDA sm_{minimum} or newer"
    return None


def legality_reason(workload, device, config):
    return None


def canonical_config(workload, config, device=None):
    return dict(config)
