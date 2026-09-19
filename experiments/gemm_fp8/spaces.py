"""The same 576 schedules and fixed scale layout on A100, H200 and B200."""

from experiments.utils.grid import grid

BLOCK_K = 128


def get_configs():
    return grid(
        block_M=[32, 64, 96, 128, 192, 256],
        block_N=[32, 64, 96, 128, 192, 256],
        block_K=[BLOCK_K],
        num_stages=[0, 1, 2, 3, 4, 5, 6, 7],
        threads=[128, 256],
    )


def support_reason(workload, device=None):
    p = workload.parameters
    if workload.dtype != "float8_e4m3fn" or not p.get("transpose_b", False):
        return "block-scaled GEMM requires E4M3 A=(M,K), B=(N,K), transpose_b=True"
    if p.get("batch", 1) != 1 or p.get("transpose_a", False) or p.get("epilogue", "none") != "none":
        return "block-scaled GEMM requires nonbatched operands and no fused epilogue"
    if p["m"] % 32 or p["n"] % 128 or p["k"] % BLOCK_K:
        return "the common aligned FP8 shape domain requires M%32=0 and N%128=K%128=0"
    if device is not None:
        from experiments.backend import FP8_COMPUTE_DTYPE

        arch = device.target.get("arch", "").removeprefix("sm_").rstrip("af")
        minimum = 80 if FP8_COMPUTE_DTYPE == "bfloat16" else 89
        if device.target["kind"] != "cuda" or not arch.isdigit() or int(arch) < minimum:
            return f"this FP8 implementation requires CUDA sm_{minimum} or newer"
    return None


def legality_reason(workload, device, config):
    return None


def canonical_config(workload, config, device=None):
    return dict(config)
