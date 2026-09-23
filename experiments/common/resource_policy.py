"""Explicit B200 compiler-resource limits, independent of memory scoring."""

# These limits come from the unfiltered 25-case E1 oracle audit.  The largest
# observed oracle allocation is 12 spill-store bytes / 8 spill-load bytes / 8
# local bytes in the two square FP8 GEMMs.  Round that family to 16 bytes and
# keep the four zero-observation families strict.  E2 is audited against these
# limits before E3 may start.
_B200_FAMILY_LIMITS = {
    "gemm": (0, 0),
    "attention": (0, 0),
    "kda_chunk_intra_token_parallel": (0, 0),
    "gemm_fp8": (16, 16),
    "grouped_gemm": (0, 0),
}


def b200_resource_limits(op):
    """Return the frozen ``(spill_bytes, local_bytes)`` limits for an op."""
    try:
        return _B200_FAMILY_LIMITS[op]
    except KeyError as error:
        raise ValueError(f"no calibrated B200 compiler-resource policy for {op!r}") from error


def b200_post_compile_policy(workload, target):
    """Return the calibrated reject-mode policy for a B200 workload.

    The operation comes from the frozen workload rather than the compiler
    filter's heuristic kernel classification.  This matters for the B200
    attention and FP8 kernels, whose legacy classifications are not reliable.
    """
    if target.get("kind") != "cuda" or target.get("arch") not in ("sm_100", "sm_100a"):
        return None
    spill, local = b200_resource_limits(workload.op)
    return dict(mode="reject", max_spill_bytes=spill, max_local_bytes=local)
