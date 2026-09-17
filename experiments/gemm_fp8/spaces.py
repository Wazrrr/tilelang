"""The FP8 example's 288 schedules expanded to 2,304 native configurations."""

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


def support_reason(workload, device=None):
    if workload.dtype not in ("float8_e4m3fn", "float8_e5m2") or not workload.parameters.get("transpose_b", False):
        return "the FP8 example requires FP8 A=(M,K), B=(N,K), transpose_b=True"
    if device is not None:
        target = device.target
        if target["kind"] == "cuda":
            import re

            match = re.fullmatch(r"sm_(\d+)[af]?", target["arch"])
            if not match or int(match[1]) < 89:
                return "native FP8 GEMM requires CUDA sm_89 or newer; Ampere has no FP8 tensor instructions"
        elif target["kind"] == "hip" and target.get("mcpu") not in ("gfx950",):
            return "this FP8 example uses OCP FP8; the selected HIP target requires a different FP8 encoding"
    return None


def legality_reason(workload, device, config):
    # Preserve the complete declared pool and record actual compiler outcomes.
    return None


def canonical_config(workload, config, device=None):
    return dict(config)
