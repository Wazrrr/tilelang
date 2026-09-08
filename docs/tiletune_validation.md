# TileTune H200 experiment findings

These are the preceding version-12 results, before the A100 portability work.
The model and generic device profile were frozen before reserved-workload
measurements. Candidate latencies were used only to evaluate ranking. Raw local
experiment directories are intentionally excluded from the source commit;
[the portable runner](../benchmark/autotune/README.md) generates a complete new
set of profiles, sources, per-config outcomes, CUDA sources and timing records.

The varied-workload experiment produced 3,072 config outcomes: four development
GEMMs with 288 configs each, plus four reserved GEMMs with 288 each and three
reserved attention workloads with 128 configs in each causal mode. It measured
2,610 correct candidates, recorded 387 compilation failures and 75 dynamic-shared
memory launch failures, with no benchmark timeouts in this run.

Version 12 added conservative shared-tile lifetime reuse and fixed-primitive
single-CTA consumer throughput limits. No per-kernel coefficients were fitted.
On the six reserved attention cases, winner retention in the first eight numeric
configs improved from 3/6 to 6/6. This does not justify a universal top-eight
cutoff: three reserved FP16/BF16 GEMMs retained only 71–94% of best performance.

| Workload | Fastest original index | v12 winner rank | Top-8 retained performance |
| --- | ---: | ---: | ---: |
| FP16 NN, 3072×6144×1536 | 191 | 12 (tie 9–16) | 87.40% |
| FP16 NT, 768×8192×4096 | 191 | 28 (tie 25–32) | 71.27% |
| BF16 TN, 6144×1536×512 | 95 | 52 (tie 49–56) | 94.16% |
| FP8 NN, 4096×2048×4096 | 186 | Unknown | Unknown |
| BSHD, B2 H12 S3072 D64, noncausal | 91 | 1 (tie 1–3) | 100% |
| BSHD, B2 H12 S3072 D64, causal | 91 | 4 (tie 4–6) | 100% |
| BSHD, B1 H8 S2048 D256, noncausal | 83 | 1 | 100% |
| BSHD, B1 H8 S2048 D256, causal | 83 | 1 | 100% |
| BHSD, B2 H4 Q1536 K3072 D128, noncausal | 91 | 1 | 100% |
| BHSD, B2 H4 Q1536 K3072 D128, causal | 91 | 1 | 100% |

FP8 NN selected MMA while the H200 generic profile measured WGMMA, so the
instruction guard correctly withheld numeric scores. Its fastest config had
4 spill-store bytes, 8 spill-load bytes and 8 local bytes. The zero-spill GEMM
policy excluded it; the fastest surviving config retained 99.42% performance.
Attention used no PTXAS spill/local cap and a 32-register logical-demand margin.

Ranking became more useful for these attention workloads, but raw unanchored
latency error worsened to about 51–65%. The model remains a ranker with incomplete
absolute timing. GEMM misses expose unmodeled cache/TMA transaction effects,
compiler copy subdivision and stage-2/stage-3 behavior. Small-tile instruction
issue limits, automatic storage reuse and compiler scratch also remain sources
of uncertainty. Neither these findings nor cross-compilation establish A100
performance; run the full grids on that device and report numeric-score coverage.
