# H200 oracle hits at larger top-K budgets

Evaluated on September 19, 2026 using the completed
[`h200-25shape-20260917T080241Z`](results/studies/h200-25shape-20260917T080241Z/REPORT.md)
study: five families, five shapes each, TileTune seed 123. These are the saved
FP16/E4M3 workloads and pools from that study, not the newer BF16/version-2
contract. The newer `h200-bf16-v2-pool7` study is incomplete and cannot yet support
an all-case comparison.

Increasing K alone cannot give TileTune all 25 exact oracle hits under the
saved selection policy. Thirteen oracle winners are `unknown` with no finite
score, so the policy excludes them at every K. XGBoost ranks every oracle
winner and reaches all 25 at K=1,684.

## Budget sweep

| Budget | TileTune exact hits | XGBoost exact hits | TileTune geomean Oracle@K | XGBoost geomean Oracle@K |
|---|---:|---:|---:|---:|
| K=20 | 12/25 | 10/25 | 91.445% | 93.960% |
| K=50 | 12/25 | 17/25 | 91.719% | 96.166% |
| K=100 | 12/25 | 20/25 | 91.719% | 97.930% |
| K=200 | 12/25 | 20/25 | 91.719% | 98.237% |
| K=500 | 12/25 | 23/25 | 91.719% | 98.858% |
| K=1,000 | 12/25 | 24/25 | 91.719% | 99.924% |
| 20% of each pool | 12/25 | 23/25 | 91.719% | 98.858% |
| 50% of each pool | 12/25 | 24/25 | 91.719% | 99.924% |
| 75% of each pool | 12/25 | 25/25 | 91.719% | 100.000% |
| 100% budget, eligible candidates only | 12/25 | 25/25 | 91.719% | 100.000% |

Percentages use `ceil(fraction * total declared pool)`. The denominator includes
failed and model-excluded configurations. The 20% budgets are 461 for GEMM and
FP8 GEMM, 64 for attention, 144 for KDA, and 39 for grouped GEMM. A requested K
above the eligible count exhausts that list and records a shortfall; it does not
append unknown candidates. Compilation and correctness failures consume budget.

An exact hit requires the saved oracle minimum latency; any exactly tied optimum
counts. Oracle@K is the oracle minimum divided by the best successful latency
within the prefix. A rounded 100% score alone is not counted as an exact hit.

## Smallest all-case cutoffs by family

| Family | Full pool per shape | TileTune reachable cases | TileTune all-hit K | XGBoost all-hit K | XGBoost smallest whole-pool percentage |
|---|---:|---:|---:|---:|---:|
| GEMM | 2,304 | 5/5 | 19 | 930 | 41% |
| FlashAttention | 320 | 1/5 | Unreachable | 37 | 12% |
| KDA | 720 | 5/5 | 5 | 29 | 4% |
| FP8 GEMM | 2,304 | 1/5 | Unreachable | 1,684 | 74% |
| Grouped GEMM | 192 | 0/5 | Unreachable | 10 | 5% |

XGBoost's last two misses at a 20% budget are `gemm_square` (oracle rank 930,
40.365% of its pool) and `gemm_fp8_square` (rank 1,684, 73.090%). Its smallest
uniform integer percentage is 74%; 75% is the first successful percentage in the
coarse sweep. The percentage cutoffs account for rounding up: for example,
`ceil(4% * 720) = 29` and `ceil(5% * 192) = 10`.

TileTune already reaches every reachable oracle by K=19. Larger budgets modestly
improve some suboptimal prefixes but cannot include the excluded winners:

- **Attention:** four winners exceed the modeled soft register allowance;
  compiler allocation and spill service remain unresolved. The short causal
  case hits at K=1. Only 65 of 320 configurations have eligible scores.
- **FP8 GEMM:** four winners lack a verified pipeline plan and have unresolved
  external-access costs. Decode hits at K=16. Eligible counts are 1,024 of 2,304,
  or 1,150 for decode.
- **Grouped GEMM:** all five winners have unresolved positive-stage pipeline
  scheduling and Hopper warp-specialization policy. Only 48 of 192 configurations
  are eligible. Even the full eligible list retains only 49.434–90.684% of oracle
  performance across these shapes.

These are recorded model uncertainties, not evidence that the oracle kernels
are invalid: the oracle successfully measured them. Achieving all hits with
TileTune requires resolving these modeling gaps or separately evaluating a
selection policy that admits unknown candidates. Merely changing `top_k` cannot
do it under this saved policy.

## Reproduce and inspect

```bash
python -m experiments.topk_study \
  --study experiments/results/studies/h200-25shape-20260917T080241Z
```

The offline analysis verifies source hashes, preserves saved ranking/tie order,
and reproduces the original Oracle@20 values. It uses the same exhaustive timing
table for every method and never retrains or reranks using oracle labels. It
does not measure larger-budget online tuning costs or fresh GPU performance.

- [Full per-case report and exclusion diagnostics](results/studies/h200-25shape-20260917T080241Z/topk-study/report.md)
- [Machine-readable results and source evidence](results/studies/h200-25shape-20260917T080241Z/topk-study/topk.json)
- [Per-case, per-budget CSV](results/studies/h200-25shape-20260917T080241Z/topk-study/topk.csv)

Validation: 25 focused tests passed for oracle comparison and top-K aggregation,
including failed-candidate budgets, tied minima, near misses, excluded winners,
percentage rounding, seed coverage, and changed-source rejection. Ruff checks
passed for the changed Python files.
