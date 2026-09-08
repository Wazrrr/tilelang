"""Workload references and evaluation for the portable TileTune experiments.

Measured candidate latencies are used only in evaluation, never in prediction.
"""

from collections import Counter
import math
from types import SimpleNamespace

from examples.flash_attention.example_mha_tiletune import make_attention
from tilelang.tiletune import TileTuneConfig, rank_records
from tilelang.tiletune.cost import combine_tile_cost
from tilelang.tiletune.ranking import apply_ranking_metric

PREFIXES = (1, 4, 8, 16, 32, 64)


def factory(workload, causal):
    batch, heads, sq, sk, dim = (workload[k] for k in ("batch", "heads", "query_length", "kv_length", "dim"))
    if workload["layout"] == "BSHD":
        assert sq == sk
        return make_attention(batch, heads, sq, dim, causal)

    def attention(block_M, block_N, num_stages, threads):
        from examples.flash_attention.example_mha_fwd_bhsd import flashattn

        return flashattn.jit_impl.get_tir(
            batch=batch,
            heads=heads,
            seq_q=sq,
            seq_kv=sk,
            dim=dim,
            is_causal=causal,
            block_M=block_M,
            block_N=block_N,
            num_stages=num_stages,
            threads=threads,
        )

    return attention


def inputs_and_reference(workload, causal):
    import torch

    generator = torch.Generator(device="cuda").manual_seed(123)
    batch, heads, sq, sk, dim = (workload[k] for k in ("batch", "heads", "query_length", "kv_length", "dim"))
    values = [torch.randn((batch, heads, seq, dim), device="cuda", dtype=torch.float16, generator=generator) for seq in (sq, sk, sk)]
    torch.backends.cuda.matmul.allow_tf32 = False
    q, k, v = [x.float() for x in values]
    reference = torch.empty_like(q)
    key_indices = torch.arange(sk, device="cuda")
    for start in range(0, sq, 256):
        end = min(start + 256, sq)
        scores = (q[:, :, start:end] @ k.transpose(-1, -2)) * dim**-0.5
        if causal:
            # BHSD example aligns unequal query/key sequences at the right end.
            query_indices = torch.arange(start, end, device="cuda") + sk - sq
            scores.masked_fill_(key_indices[None, :] > query_indices[:, None], -float("inf"))
        reference[:, :, start:end] = scores.softmax(-1) @ v
    if workload["layout"] == "BSHD":
        values = [x.permute(0, 2, 1, 3).contiguous() for x in values]
        reference = reference.permute(0, 2, 1, 3).contiguous()
    return values, reference


def traffic_ranking(records):
    rows = []
    for record in records:
        modules = record.get("modules") or {}
        score = None
        if "memory_traffic" in modules and "waves" in modules and "traffic_bytes_per_block" in modules["memory_traffic"]:
            cost = combine_tile_cost(modules["memory_traffic"], modules["waves"])
            result = apply_ranking_metric(
                cost,
                modules["waves"],
                {},
                TileTuneConfig(ranking_metric="traffic_waves"),
                SimpleNamespace(matched=record["specialization"]["matched"]),
                record["pressure"].get("register_demand"),
            )
            score = result["score"]
        rows.append(
            dict(
                index=record["index"], pre_lowering=record.get("pre_lowering"), tile_cost=dict(score=score, ranking_metric="traffic_waves")
            )
        )
    return rank_records(rows)


def evaluate_ranking(ranking, records, measured):
    from scipy.stats import spearmanr

    valid = {
        r["index"]: r["latency_ms"]
        for r in measured
        if r["benchmark_status"] == "ok" and math.isfinite(r["latency_ms"]) and r["latency_ms"] > 0
    }
    entries = {r["index"]: r for r in ranking}
    if not valid:
        return dict(correct=0, tiers=dict(Counter(r["tier"] for r in ranking)))
    winner = min(valid, key=valid.get)
    entry = entries[winner]
    scored = [r["index"] for r in ranking if r["tier"] == "eligible" and r["index"] in valid]
    numeric = [r for r in ranking if r["tier"] == "eligible"]
    prefix = {}
    for count in PREFIXES:
        chosen = [r["index"] for r in numeric[:count] if r["index"] in valid]
        best = min(chosen, key=valid.get) if chosen else None
        prefix[str(count)] = dict(best_index=best, retained_percent=100 * valid[winner] / valid[best] if best is not None else None)
    result = dict(
        correct=len(valid),
        numeric_correct=len(scored),
        tiers=dict(Counter(r["tier"] for r in ranking)),
        winner=dict(
            index=winner,
            config=measured[winner]["config"],
            latency_ms=valid[winner],
            tier=entry["tier"],
            rank=entry["rank"] if entry["tier"] == "eligible" else None,
            report_position=entry["rank"],
            tie_first=entry["tie_first_rank"],
            tie_last=entry["tie_last_rank"],
        ),
        spearman=float(spearmanr([entries[i]["score"] for i in scored], [valid[i] for i in scored]).statistic) if len(scored) > 1 else None,
        prefix=prefix,
    )
    if result["spearman"] is not None and not math.isfinite(result["spearman"]):
        result["spearman"] = None
    rejected = [i for i in valid if any((records[i].get(stage) or {}).get("would_reject") for stage in ("pre_lowering", "post_compile"))]
    kept = [i for i in valid if i not in rejected]
    result.update(
        correct_removed_by_rejection=rejected,
        rejection_retained_percent=100 * valid[winner] / min(valid[i] for i in kept) if kept else None,
    )
    return result
