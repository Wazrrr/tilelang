"""Explicit H200 compiler-resource limits, independent of memory scoring."""


def h200_post_compile_policy(workload, target):
    """Keep calibrated spill allowances at the experiment boundary.

    Fresh compilation of the frozen 25-case H200 oracle configs observed up to
    48-byte spills in attention and 92-byte spill stores in FP8 GEMM. Round the
    limits up to 64 and 128 bytes; the other families' winners had zero spills.
    These are PTXAS counters, not estimates of runtime spill traffic. The same
    bounds apply to local/stack bytes. Hardware limits are never relaxed.
    """
    if target.get("kind") != "cuda" or target.get("arch") not in ("sm_90", "sm_90a"):
        return None
    allowance = {"attention": 64, "gemm_fp8": 128}.get(workload.op, 0)
    return dict(mode="reject", max_spill_bytes=allowance, max_local_bytes=allowance)
