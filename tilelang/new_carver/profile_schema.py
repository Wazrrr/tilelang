"""Accepted primitive rates, profile metadata and validation.

The schema is shared by every kernel family. Device measurements are produced
by device_profile.py; validation never profiles a device or a candidate kernel.
"""

from math import isfinite


RATE_FIELDS = {
    "global_bytes_per_cycle",
    "shared_bytes_per_cycle",
    "gemm_flops_per_cycle",
    "wgmma_flops_per_cycle_per_warpgroup",
    "latency_scale",
    "reference_clock_mhz",
    "elementwise_ops_per_cycle",
    "exp_ops_per_cycle",
    "reduction_ops_per_cycle",  # Legacy profile field; not used for mapped tile reductions.
    "reduction_local_sum_per_cycle",
    "reduction_local_max_per_cycle",
    "reduction_shuffle_sum_per_cycle",
    "reduction_shuffle_max_per_cycle",
}
LATENCY_FIELDS = {"copy_latency_cycles", "barrier_cycles"}
CONSUMER_RATE_FIELDS = {
    "elementwise_ops_per_cycle",
    "exp_ops_per_cycle",
    "reduction_local_sum_per_cycle",
    "reduction_local_max_per_cycle",
    "reduction_shuffle_sum_per_cycle",
    "reduction_shuffle_max_per_cycle",
}
PROFILE_METADATA_FIELDS = {"gemm_signature", "profile_target", "profile_id", "memory_regime", "reduction_dtype"}


def validate_performance_model(profile):
    if not isinstance(profile, dict) or set(profile) - RATE_FIELDS - LATENCY_FIELDS - PROFILE_METADATA_FIELDS - {"consumer_rates"}:
        raise ValueError("performance_model contains unsupported fields")
    for key, value in profile.items():
        if key == "consumer_rates":
            if not isinstance(value, dict) or not value:
                raise ValueError("consumer_rates requires measured thread-count rows")
            for threads, rates in value.items():
                if not isinstance(threads, str) or not threads.isdigit() or not 0 < int(threads) <= 1024 or int(threads) % 32:
                    raise ValueError("consumer_rates keys must be positive warp-multiple thread counts")
                if not isinstance(rates, dict) or not rates or set(rates) - CONSUMER_RATE_FIELDS:
                    raise ValueError("consumer_rates contains unsupported primitives")
                validate_performance_model(rates)
            continue
        if key in PROFILE_METADATA_FIELDS:
            if key == "gemm_signature":
                if (
                    not isinstance(value, dict)
                    or set(value) != {"instruction", "a_dtype", "b_dtype", "accum_dtype"}
                    or not all(isinstance(v, str) and v for v in value.values())
                ):
                    raise ValueError("gemm_signature requires instruction, a_dtype, b_dtype and accum_dtype strings")
            elif key == "memory_regime" and value not in ("cached", "streaming"):
                raise ValueError("memory_regime must be cached or streaming")
            elif not isinstance(value, str) or not value:
                raise ValueError(f"{key} must be a nonempty string")
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, float | int)
            or not isfinite(value)
            or value < 0
            or (key in RATE_FIELDS and value == 0)
        ):
            raise ValueError("performance_model requires finite positive rates and nonnegative latencies")
