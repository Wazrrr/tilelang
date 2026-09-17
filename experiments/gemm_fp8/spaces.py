"""2,304 native schedules, containing the FP8 example's 288-config grid."""

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
    if workload.dtype not in ("float8_e4m3fn", "float8_e5m2"):
        return "FP8 GEMM requires float8_e4m3fn or float8_e5m2"
    if not workload.parameters.get("transpose_b", True):
        return "the FP8 example uses A=(M,K), B=(N,K), transpose_b=True"
    return None


def legality_reason(workload, device, config):
    return None


def canonical_config(workload, config, device=None):
    return dict(config)
