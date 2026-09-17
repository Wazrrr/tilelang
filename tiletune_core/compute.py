"""Service accounting on compiler-resolved work counts."""

from .profile_schema import CONSUMER_RATE_FIELDS, reduction_rate_field


def estimate_phase_cycles(phase, profile, concurrent_ctas):
    """Apply aggregate SM rates and optional consumer/warpgroup ceilings."""
    terms = {}

    def service(amount, rate_key):
        if not amount:
            return 0
        rate = profile.get(rate_key)
        if not rate:
            return None
        aggregate = amount * concurrent_ctas / rate
        if rate_key not in CONSUMER_RATE_FIELDS or not profile.get("consumer_rates"):
            return aggregate
        # A CTA cannot use all SM issue capacity when too few consumer
        # warps are ready. Producer warps do not execute these operations.
        row = profile["consumer_rates"].get(str(phase.get("consumer_threads")), {})
        single = row.get(rate_key)
        return max(aggregate, amount / single) if single else None

    for key, amount in phase["work"].items():
        if key == "reduction_ops":
            terms[key] = 0
            continue
        if amount is None:
            return None
        rate_key = {
            "gemm_flops": "gemm_flops_per_cycle",
            "shared_bytes": "shared_bytes_per_cycle",
            "elementwise_ops": "elementwise_ops_per_cycle",
            "exp_ops": "exp_ops_per_cycle",
            "rsqrt_ops": "rsqrt_ops_per_cycle",
            "log_ops": "log_ops_per_cycle",
            "convert_float32_to_float8_e4m3fn": "convert_float32_to_float8_e4m3fn_per_cycle",
            "convert_float32_to_float8_e5m2": "convert_float32_to_float8_e5m2_per_cycle",
            "reduction_ops": "reduction_ops_per_cycle",
        }[key]
        if amount and not profile.get(rate_key):
            return None
        terms[key] = service(amount, rate_key)
        if terms[key] is None:
            return None
    if phase["work"]["reduction_ops"]:
        reduction = phase.get("reduction") or {}
        if reduction.get("precision") != "predicted":
            return None
        for operation in ("local", "shuffle"):
            amount = reduction[f"{operation}_pairs"]
            field = reduction_rate_field(reduction, profile, operation)
            rate = profile.get(field)
            if amount and not rate:
                return None
            cycles = service(amount, field)
            if cycles is None:
                return None
            terms["reduction_ops"] += cycles
        if reduction.get("shared_pairs"):
            pairs = reduction["shared_pairs"]
            combine = service(pairs, reduction_rate_field(reduction, profile, "local"))
            element_bytes = reduction.get("element_bytes")
            if element_bytes is None:
                # Archived facts supported scalar reductions only.
                element_bytes = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8, "int32": 4, "int64": 8}.get(reduction["dtype"])
            if element_bytes is None:
                return None
            shared = service(pairs * 2 * element_bytes, "shared_bytes_per_cycle")
            if combine is None or shared is None or profile.get("barrier_cycles") is None:
                return None
            terms["reduction_ops"] += (
                combine + shared + (reduction["barrier_rounds"] + reduction.get("workspace_reuse_barriers", 0)) * profile["barrier_cycles"]
            )
    # Matrix instructions consume tensor-core and shared-memory service;
    # scalar/reduction/exp phases execute in program order.
    group_service = 0
    group_rate = profile.get("wgmma_flops_per_cycle_per_warpgroup")
    if group_rate and phase["work"]["gemm_flops"]:
        participants = phase.get("compute_participants") or {}
        if participants.get("precision") != "predicted":
            return None
        if participants["instruction"] == "cuda.wgmma":
            groups = participants.get("warpgroups")
            if not groups:
                return None
            group_service = phase["work"]["gemm_flops"] / (groups * group_rate)
    return (
        max(terms["gemm_flops"], terms["shared_bytes"], group_service)
        + terms["elementwise_ops"]
        + terms["exp_ops"]
        + terms.get("rsqrt_ops", 0)
        + terms.get("log_ops", 0)
        + terms.get("convert_float32_to_float8_e4m3fn", 0)
        + terms.get("convert_float32_to_float8_e5m2", 0)
        + terms["reduction_ops"]
    )
