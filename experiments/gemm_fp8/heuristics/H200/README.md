# H200 FP8 oracle references

Each workload JSON contains a full-oracle path and SHA-256, all outcome counts,
the winning configuration, and seven fresh winner measurements. Consult that
reference and its source/compiler provenance before using a saved result.

Records pointing to `fp8-matrix-20260917T011418Z` used the earlier WGMMA-enabled
compiler policy. That sweep attempted all 2,304 candidates per final shape;
its 32-row MMA configurations passed, while WGMMA configurations failed the
strict numerical check. Those records are historical and do not establish
performance for the current experiment-wide `tl.disable_wgmma=True` policy.

The [fresh study](../../../results/studies/fp8-strict-mma-20260917T0320Z/RUN.md)
collects new measurements under the explicit MMA policy. Its completed exports
will update these references; previous full oracles remain in their study
directories. A queued study or winner-only record is not a completed baseline
comparison.
