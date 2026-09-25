"""Carver adapter for the token-parallel KDA intra kernel.

The intra kernel issues no tensor-core MMA, so this adapter scores the kernel's
own CTA domain and gated QK/KK memory ledger instead of a tensorized GEMM.
"""


def support_reason(workload, device=None):
    if device is not None and device.target.get("kind") != "cuda":
        return "the Carver KDA adapter requires CUDA"
    expected = {"batch", "heads", "sequence", "dim", "chunk_size", "sub_chunk_size"}
    if workload.op != "kda_chunk_intra_token_parallel" or not expected <= set(workload.parameters):
        return "the Carver KDA adapter supports the token-parallel intra-chunk kernel only"
    return None


def carver_rank(workload, device, configs, top_k):
    reason = support_reason(workload, device)
    if reason:
        raise ValueError(reason)
    from experiments.common.carver import kda_intra_rank

    report = kda_intra_rank(workload, device, configs, top_k)
    report["model"] = "carver_kda_intra_traffic_waves"
    return report
