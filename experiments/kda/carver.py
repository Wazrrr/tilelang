"""Carver adapter for token-parallel KDA intra coefficients."""

import math


def support_reason(workload, device):
    if device.target["kind"] != "cuda":
        return "the KDA Carver adapter requires CUDA"
    if workload.op != "kda_chunk_intra_token_parallel" or workload.dtype not in ("float16", "bfloat16"):
        return "the KDA Carver adapter supports FP16/BF16 token-parallel intra coefficients only"
    return None


def carver_rank(workload, device, configs, top_k):
    reason = support_reason(workload, device)
    if reason:
        raise ValueError(reason)

    from experiments.common.carver import _architecture, _occupancy, _rank_records, workload_template

    p = workload.parameters
    arch = _architecture(device.target)
    template = workload_template(workload, configs, arch=arch)
    batch, heads, sequence, key_dim, chunk, sub_chunk = (
        p[key] for key in ("batch", "heads", "sequence", "dim", "chunk_size", "sub_chunk_size")
    )
    average_iterations = (sub_chunk + 1) / 2

    def evaluate(config):
        block_heads = config["block_H"]
        stages = max(1, config["num_stages"])
        grid = batch * sequence * math.ceil(heads / block_heads)

        fixed_load_bytes = block_heads * (8 * key_dim + 2)
        loop_load_bytes = average_iterations * block_heads * 6 * key_dim
        output_bytes = block_heads * 2 * (chunk + sub_chunk)
        traffic = fixed_load_bytes + loop_load_bytes + output_bytes

        fixed_shared_bytes = block_heads * (16 * key_dim + 2 + 2 * (chunk + sub_chunk))
        staged_shared_bytes = stages * block_heads * 6 * key_dim
        shared_bytes = fixed_shared_bytes + staged_shared_bytes
        register_words = block_heads * (3 * key_dim + 2)
        valid, blocks_per_sm, waves = _occupancy(
            arch,
            grid_blocks=grid,
            shared_bytes=shared_bytes,
            register_words=register_words,
            threads=config["threads"],
        )
        return dict(
            valid=valid,
            traffic_bytes_per_cta=traffic,
            shared_bytes=shared_bytes,
            register_words=register_words,
            grid_blocks=grid,
            blocks_per_sm=blocks_per_sm,
            waves=waves,
            average_loop_iterations=average_iterations,
        )

    report = _rank_records(configs, top_k, arch=arch, template=template, evaluate=evaluate)
    report.update(
        metric="carver_traffic_waves",
        score_units="byte-waves",
        formula="(estimated global bytes per CTA + 1) * occupancy waves",
    )
    report["assumptions"].extend(
        [
            "The model follows the token-parallel coefficient kernel, not the separate KDA chunk-output kernel.",
            "Loop traffic uses the exact average causal sub-chunk length; boundary head tiles retain full-tile cost.",
            "Pipeline depth scales the staged K/G shared-memory buffers, with stages zero represented by one buffer.",
        ]
    )
    return report
