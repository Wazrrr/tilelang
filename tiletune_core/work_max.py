"""Fixed-rate memory/compute fusion, independent of the candidate pool."""

from fractions import Fraction
from math import ceil

from .memory import _memory_order, score_memory
from .profile_schema import validate_performance_model


WORK_KINDS = (
    "gemm_flops",
    "tcgen05_gemm_flops",
    "elementwise_ops",
    "exp_ops",
    "rsqrt_ops",
    "reduction_ops",
    "reduction_max_ops",
)


def score_work_max(accesses, compute, grid_blocks, sm_count, performance_model=None, pipeline_depth=1):
    """Order max(memory cycles, summed compute service cycles) at fixed rates.

    Rates are logical work per SM per cycle, not chip-wide nominal peaks.
    Convert compute service to memory-equivalent bytes, rounding upward by less
    than one byte, to retain exact integer ordering and the memory tie policy.
    No calibration, pool normalization, occupancy gate or measured candidate
    latency is used here. Missing nonzero-work rates leave the score unknown.
    """
    profile = performance_model if performance_model is not None else {}
    validate_performance_model(profile)
    memory = score_memory(accesses, grid_blocks, sm_count, pipeline_depth)
    unknown = [*memory["unknown"], *compute.get("unknown", [])]
    work = compute.get("work_per_cta", {})
    if set(work) - set(WORK_KINDS):
        raise ValueError("unsupported compute work kind")
    service = {}
    for kind in WORK_KINDS:
        amount = work.get(kind)
        if amount is None:
            unknown.append(f"unresolved {kind}")
        elif type(amount) is not int or amount < 0:
            raise ValueError(f"{kind} must be a nonnegative integer or None")
        elif amount == 0:
            service[kind] = Fraction(0)
        elif not profile.get(f"{kind}_per_cycle"):
            unknown.append(f"missing profile rate {kind}_per_cycle")
        else:
            service[kind] = Fraction(amount) / Fraction(str(profile[f"{kind}_per_cycle"]))
    bandwidth = profile.get("global_bytes_per_cycle")
    if not bandwidth:
        unknown.append("missing profile rate global_bytes_per_cycle")
    if work.get("gemm_flops") or work.get("tcgen05_gemm_flops"):
        signature = profile.get("gemm_signature")
        signatures = compute.get("matrix_signatures")
        if not signature or not signatures:
            unknown.append("matrix work requires a matching GEMM instruction/dtype signature")
        else:
            for actual in signatures:
                matches = all(actual.get(key) == signature[key] for key in ("a_dtype", "b_dtype", "accum_dtype"))
                instruction = actual.get("instruction", "")
                matches &= instruction == signature["instruction"] or (
                    instruction == "cuda.tcgen05" and bool(profile.get("tcgen05_gemm_flops_per_cycle"))
                )
                if not matches:
                    unknown.append("device profile GEMM instruction/dtype signature does not match the kernel")
    if (work.get("reduction_ops") or work.get("reduction_max_ops")) and compute.get("reduction_dtypes") != [
        profile.get("reduction_dtype", "float32")
    ]:
        unknown.append("device profile reduction dtype does not match the kernel")
    if compute.get("target_arch") and profile.get("profile_target") not in (None, compute["target_arch"]):
        unknown.append("device profile target does not match the kernel")
    waves = memory["single_cta_waves"]
    compute_cycles = sum(service.values(), Fraction(0)) * waves if waves is not None and not unknown else None
    memory_cycles = Fraction(memory["logical_byte_waves"]) / Fraction(str(bandwidth)) if not unknown else None
    compute_bytes = ceil(compute_cycles * Fraction(str(bandwidth))) if not unknown else None
    effective_bytes = max(memory["logical_byte_waves"], compute_bytes) if not unknown else None
    score = _memory_order(effective_bytes, pipeline_depth, memory["logical_memory_access_waves"]) if not unknown else None
    return {
        **memory,
        "metric": "work_max",
        "score": score,
        "tie_break_score": memory["tie_break_score"] if not unknown else None,
        "units": "lexicographic memory-equivalent work units",
        "formula": "lexicographic(max(byte_waves, ceil(bandwidth * compute_cycles)), -pipeline_depth, access_waves)",
        "compute_formula": "ceil(grid_blocks / sm_count) * sum(work_per_cta[kind] / rate[kind])",
        "adjusted_logical_byte_waves": effective_bytes,
        "compute_equivalent_byte_waves": compute_bytes,
        "compute_work_per_cta": dict(work),
        "service_cycles": {
            "memory": float(memory_cycles) if memory_cycles is not None else None,
            "compute": float(compute_cycles) if compute_cycles is not None else None,
        },
        "compute_cycles_by_kind": {kind: float(value * waves) for kind, value in service.items()} if not unknown else {},
        "max_service_cycles": float(max(memory_cycles, compute_cycles)) if not unknown else None,
        "performance_model": dict(profile),
        "occupancy_gate_enabled": False,
        "precision": "unknown" if unknown else "estimate",
        "unknown": sorted(set(unknown)),
        "assumptions": [
            "fixed logical-work rates per SM per cycle; no candidate-pool normalization or latency fitting",
            "padded and predicated logical work with single-CTA waves; no physical occupancy estimate",
            "compute service terms sum; memory and compute service overlap by max",
            "global bytes are logical requests, not measured HBM transactions; the memory regime is explicit in the profile",
            "logical reductions use sum/max primitive rates, not an inferred physical lane/shuffle schedule",
            "no cache-capacity, synchronization, shared-memory or tensor-memory transfer timing",
            "compute-to-memory normalization rounds upward by less than one equivalent byte using exact rational arithmetic",
            "equal complete keys remain tied; pipeline depth and access count only order equal normalized work",
        ],
    }


def score_work_rank_product(accesses, compute, grid_blocks, sm_count, performance_model=None, pipeline_depth=1):
    """Prepare fixed-rate work_max and underfill views for pool-scoped fusion."""
    accesses = tuple(accesses)
    work = score_work_max(accesses, compute, grid_blocks, sm_count, performance_model, pipeline_depth)
    underfill = score_memory(accesses, grid_blocks, sm_count, pipeline_depth, launch_underfill=True)
    return {
        **work,
        "metric": "work_rank_product",
        "score": None,
        "tie_break_score": None,
        "score_scope": "candidate_pool",
        "component_scores": {"work_max": work["score"], "underfill": underfill["score"]},
        "underfill_logical_byte_waves": underfill["adjusted_logical_byte_waves"],
        "units": "squared candidate tail ranks",
        "formula": "tail_rank(work_max) * tail_rank(underfill)",
        "assumptions": work["assumptions"]
        + [
            "both component tail ranks use the same eligible candidate pool",
            "rank_records recomputes the fused score for each pool; no candidate latency enters ranking",
            "underfill adjusts only its own memory view, never the work_max compute component",
        ],
    }
