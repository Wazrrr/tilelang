# Benchmark contract (B200)

The final suite has five kernel families and five holdout shapes per family
(25 cases total). The shape count stays compact while covering every requested
axis:

- GEMM: BF16 A/B/C, FP32 accumulation; M=256 small-M, square, rectangular,
  and K=14336 long-reduction cases.
- FP8 GEMM: E4M3 A/B, explicit per-row/per-128-K-block packed UE8M0 scales,
  FP32 accumulation, and BF16 C. It uses the SM100
  `gemm_mxfp8_blockscaled_1d1d.py` TCGen05 two-CTA tiled and persistent kernels.
  The eight candidates share a fixed 128x256x128 tile and vary persistent
  scheduling, swizzle size, and store policy.
- Grouped GEMM: BF16 A/B/C and FP32 accumulation. The five cases vary 3, 4,
  and 8 groups, balanced versus skewed row counts, and both projection
  directions. This is deliberately the BF16 grouped example, not the separate
  grouped-MXFP8 kernel.
- FlashAttention: BF16 forward with FP32 online softmax and accumulation. The
  cases vary sequence length, head dimension, batch size, head count, and
  causal masking while using the SM100 attention example.
- KDA: only `examples/kda/chunk_o.py`, with BF16 Q/V/A/state and BF16 output,
  the example's fixed FP32 gate semantics, FP32 component accumulations,
  DK=DV=128, chunk size 64, and varying sequence, batch, and head count.

Training uses two independent workloads per family and validation uses one;
the final holdouts are not used to fit XGBoost. Brute force, Carver, XGBoost,
and TileTune consume the same family pool. Source hashes include the selected
example, adapter, reference, and configuration space, so the old FP8-output and
grouped-MXFP8 measurements cannot be reused for this contract.

This file defines workloads and implementations, not performance results. A
result is valid only when the monitored runner records an idle matching GPU and
no contention for the measurement interval.
