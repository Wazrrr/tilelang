# Benchmark contract (A100)

The final suite has five kernel families and five holdout shapes per family
(25 cases total). The shape count stays compact while covering every requested
axis:

- GEMM: BF16 A/B/C, FP32 accumulation; M=256 small-M, square, rectangular,
  and K=14336 long-reduction cases.
- FP8 GEMM: E4M3 storage, explicit per-row/per-128-K-block scales, FP32
  accumulation, and BF16 C. A100 has no native FP8 tensor instruction, so
  `example_mxfp8_blockscaled_gemm_a100.py` converts E4M3 tiles to BF16 in shared
  memory and uses Ampere BF16 tensor cores. The 72-candidate pool tunes that
  emulation; reports must not describe it as native FP8 throughput.
- Grouped GEMM: BF16 A/B/C and FP32 accumulation. The five cases vary 3, 4,
  and 8 groups, balanced versus skewed row counts, and both projection
  directions.
- FlashAttention: BF16 forward with FP32 online softmax and accumulation. The
  cases vary sequence length, head dimension, batch size, head count, and
  causal masking.
- KDA: only `examples/kda/chunk_o.py`, with BF16 Q/V/A/state and BF16 output,
  the example's fixed FP32 gate semantics, FP32 component accumulations,
  DK=DV=128, chunk size 64, and varying sequence, batch, and head count.

Training uses two independent workloads per family and validation uses one;
the final holdouts are not used to fit XGBoost. Brute force, Carver, XGBoost,
and TileTune consume the same family pool. Source hashes include the selected
example, adapter, reference, and configuration space, so native-FP8 or old
FP8-output measurements cannot be reused for this contract.

This file defines workloads and implementations, not performance results. A
result is valid only when the monitored runner records an idle matching GPU and
no contention for the measurement interval.
