"""Shared Carver traffic/wave adapter including scales and both accumulators."""

import math


def carver_rank(workload, device, configs, top_k):
    from experiments.backend import FP8_COMPUTE_DTYPE
    from experiments.common.carver import _architecture, _occupancy, _rank_records, workload_template

    arch = _architecture(device.target)
    template = workload_template(workload, configs, arch=arch)
    m, n, k = (workload.parameters[key] for key in ("m", "n", "k"))
    compute_bytes = 2 if FP8_COMPUTE_DTYPE == "bfloat16" else 1

    def evaluate(c):
        bm, bn, bk = (c[key] for key in ("block_M", "block_N", "block_K"))
        iterations = k // 128
        grid = math.ceil(m / bm) * math.ceil(n / bn)
        operand_bytes = (bm + bn) * k  # E4M3 global storage, including Ampere.
        scale_bytes = 4 * (bm + bn) * iterations
        output_bytes = 2 * bm * bn
        shared = max(1, c["num_stages"]) * (bm + bn) * (bk * compute_bytes + 4) + output_bytes
        registers = 2 * bm * bn  # FP32 partial and accumulated products.
        valid, blocks, waves = _occupancy(arch, grid_blocks=grid, shared_bytes=shared, register_words=registers, threads=c["threads"])
        return dict(
            valid=valid,
            traffic_bytes_per_cta=operand_bytes + scale_bytes + output_bytes,
            operand_bytes=operand_bytes,
            scale_bytes=scale_bytes,
            output_bytes=output_bytes,
            shared_bytes=shared,
            register_words=registers,
            grid_blocks=grid,
            blocks_per_sm=blocks,
            waves=waves,
            scale_flops=3 * bm * bn * iterations,
        )

    report = _rank_records(configs, top_k, arch=arch, template=template, evaluate=evaluate)
    report["model"] = "carver_blockscaled_common_grid_v1"
    report["assumptions"] = [
        "The same traffic/wave formula is used on all three CUDA architectures.",
        "Global operands are E4M3; FP32 scales use one value per row and 128 K elements for both A and B.",
        "Shared operands use BF16 on Ampere and native E4M3 otherwise; both FP32 accumulator tiles are charged.",
        "Scale arithmetic is recorded but not assigned a latency; compiler legality and tensor-core timing are not predicted.",
        "No oracle measurements enter this ranking; ties keep the common pool order.",
    ]
    return report
