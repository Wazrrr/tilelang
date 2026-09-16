# Expanded kernel configuration spaces

The portable suite now has `current`, `expanded`, and `large` presets. The
current preset retains the existing grids and indices, including the tiled KDA
implementations. The larger presets add actual scheduling choices and keep the
smaller preset's candidates as a prefix. Explicit configuration lists continue
to take precedence.

For the default FP16 shapes on A100:

| Kernel | Current | Expanded | Large |
| --- | ---: | ---: | ---: |
| GEMM and variants | 108 | 2,060 | 6,180 |
| Attention and causal attention | 54 | 1,044 | 5,004 |
| Recurrent KDA | 114 | 1,326 | 1,656 |
| Chunk-output KDA | 420 | 3,336 | 7,384 |
| Softmax / RMSNorm / row sum | 6 each | 409 each | 691 each |
| Elementwise | 6 | 279 | 399 |

These are declared candidate counts after structural constraints and known
aliases, before compiler/resource failures and generated-code equivalence
checks. They are not assertions that every candidate is executable or unique
after compilation.

## Implemented scheduling choices

- GEMM: wider M/N/K tiles and pipeline depths, three warp policies, and CTA
  traversal panels. Fused bias loads now guard the N tail.
- Attention: independent QK and PV warp policies, copy width, wider tiles and
  stages, and a separate tiled implementation with explicit softmax ownership.
  Shared intermediates convert between the three layouts. Its pipelined causal
  path uses a fixed full-KV loop with masking to keep prefetch/drain indexing
  correct for short prefixes; its additional work is included in timing. The
  zero-stage path stops at the causal prefix. The
  original direct attention implementation remains available.
- KDA: wider ranges for the existing independent token/key/value/row/causal
  tiles, unroll factors, pipeline depths and thread counts. Recurrence order,
  final state, gate conventions and chunk boundaries are preserved. A larger
  development sweep exposed a short-prefix pipeline bug in chunk output.
  Pipelined intra-chunk work now uses a fixed full-chunk bound with masking;
  the serial path retains its shorter causal prefix. The extra work is measured.
- Row reductions: streamed column tiles with complete-row softmax/RMS statistics,
  explicit row-group ownership and vector widths. Both passes are measured.
- Elementwise: independent row/column tiling and explicit vector/thread layouts.

[`spaces.py`](../experiments/common/spaces.py) records generated choices,
rejection reasons, aliases and stable configuration IDs. Its Ampere warp-policy
alias rules are checked against the compiler's policy implementation. Unknown
register pressure and unsupported TileTune scores do not narrow the shared pool.
The compiler remains the authority on resource feasibility.

## Training and experiment infrastructure

XGBoost still uses the seeded **10%** subset of each training and validation
pool. Failed samples consume the budget and are not replaced. Its feature schema
now comes from the declared, unlabeled training pools, so new implementation
parameters are recognized even when they have no successful sampled labels.
Only sampled successful measurements enter the training matrix. Version 2
artifacts record this schema source; version 1 artifacts remain readable with
their original schema and execution-domain checks.

The baseline retains 600 maximum boosting rounds, depth 10, learning rate 0.05,
row subsampling 0.8 and validation patience 20. TileTune uses `pipeline_time`.
Online K remains separate from training sampling. `compare --methods` freezes
the requested methods before collection, and the independent exhaustive oracle
runs last. Diagnostics include K=1/5/10/20/50 curves from frozen rankings.

`python -m experiments.common.census` runs a checkpointed compilation and
correctness census in isolated shards. It records generated device-source hashes,
compile/measurement failures, and current-pool versus expanded-pool best timings.
It preserves interrupted shards and checks plan/source/compiler identities on
resume. Source identity does not establish binary equivalence. Compilation costs
are retained rather than supplied free to either selector.

## Validation

The broad TileTune/experiment regression run passed **464 tests**, with 13
skipped and 3 Blackwell tests deselected. After the KDA repair, 85 expanded-kernel
and existing KDA tests passed. A final attention run passed 16 checks, including
head dimensions 32/64/128 and the zero-stage causal shortcut. The new kernel test
file now has 43 checks. The focused planning, census, XGBoost and comparison run
passed 63 tests. These runs overlap; their counts should not be added together.

Targeted development censuses checked the following samples. Selection uses
declared configuration values without latency inputs, with seven additional KDA
regressions from the aborted collection. The final attention sample covers every
declared value on each expanded axis.

| Family | Compiled | Correct | Distinct correct device sources |
| --- | ---: | ---: | ---: |
| GEMM | 32 | 30 | 30 |
| Causal attention, final implementation | 32 | 26 | 26 |
| Recurrent KDA | 32 | 32 | 32 |
| Chunk-output KDA, final implementation | 39 | 36 | 36 |
| Softmax | 32 | 32 | 32 |
| RMSNorm | 32 | 32 | 32 |
| Row sum | 32 | 32 | 32 |
| Elementwise | 32 | 32 | 32 |

The eleven unsuccessful executions exceeded A100's allowed dynamic shared memory;
they did not produce accepted timings. All final sampled configs compiled. The
attention and KDA reruns replace earlier development results that exposed layout
and short-prefix pipeline problems. These were addressed before the final pilot.
This targeted sample does not establish full-pool success rates.

Records are in the [development census](../experiments/results/expanded-development-census-20260914),
[final attention census](../experiments/results/expanded-attention-census-v2-20260914),
and [final KDA census](../experiments/results/expanded-kda-census-final-20260914).
A real three-config, two-shard softmax run also verified that resume preserves
shard files and keeps original config indices unique in the merged census.

## Full-pool pilot

The final pilot evaluates GEMM, causal attention, chunk-output KDA and softmax on
A100, using the complete expanded pools. Training scales are 0.25/0.5,
validation is 0.75, and the held-out test scale is 1.25. Both methods have a
maximum of 20 online trials. It uses one sampling seed (123), 32 compilation
workers, warmup 5, rep 20 and seven shuffled winner remeasurements. This is a
pilot, not the multi-seed final study proposed in the original plan.
CUDA-event measurements flush L2 before each measured invocation; the flush
itself is outside the kernel timing interval.
The retained-current-grid comparison uses the same final kernel implementations
as the expanded grid, including correctness repairs. It isolates the benefit of
additional schedule choices rather than comparing against an older buggy source.

The [first collection](../experiments/results/expanded-pilot-20260914) was stopped
before test collection when seven sampled KDA configs failed numerical checks.
Its logs and original source archive are preserved. The pipeline fix was tested
before restarting in a [new result directory](../experiments/results/expanded-pilot-v2-20260914).
The fixed primitive profile is reusable because its probes, hardware and compiler
are unchanged; its original 195.70-second preparation cost is retained. Training
and validation labels are recollected under the final source fingerprints.
All training and validation collection is complete. The models use 1,278
training labels and 635 validation labels, from 1,372 and 686 attempted configs
respectively. Those attempts come from declared pools totaling 13,698 training
and 6,849 validation candidates across shapes. There is no replacement sampling.

| Model | Training labels / attempts | Validation labels / attempts | Best boosting rounds | Collection + fitting |
| --- | ---: | ---: | ---: | ---: |
| GEMM | 379 / 412 | 188 / 206 | 322 | 7.15 min |
| Attention | 163 / 210 | 77 / 105 | 438 | 21.13 min |
| Chunk-output KDA | 654 / 668 | 329 / 334 | 277 | 15.95 min |
| Softmax | 82 / 82 | 41 / 41 | 309 | 1.63 min |

These collection costs use the worker's tuning timer; process startup and input
preparation are separately available in `worker_wall_seconds`. Fitting itself
took 2.22 seconds across the four models. The models and both selectors' rankings
are frozen before their held-out oracles. The complete source archive includes
122 Python files, including the attention example dependencies recorded by workers.

The attention model has a measurable training-coverage gap: the two training
samples attempted only three retained-baseline configs, and all three failed
compilation. All 163 successful training labels therefore describe the new
tiled implementation. Validation contains four successful baseline examples,
but validation labels do not fit trees. The
[implementation coverage audit](../experiments/results/expanded-pilot-v2-20260914/attention-training-coverage.json)
records this limitation of this seed's uniform sampling. The schema correctly
accepts the unsampled implementation; that does not supply performance examples.
Implementation-stratified sampling under the same fixed budget is a useful
subsequent experiment, not a change made to this frozen baseline.

All four held-out sweeps and both rounds of winner validation are complete.
The final test shapes are GEMM M=N=K=1,280; causal attention B=1, H=4,
S=1,280, D=64; chunk-output KDA B=1, H=4, S=1,280, K=V=128,
chunk size 128; and softmax 1,280 rows by 2,560 columns.

Across the four pools, **6,310 candidates passed correctness**. The failures
were 512 dynamic-shared-memory limits (156 GEMM, 280 attention, 76 KDA)
and 27 attention compilation failures, all in the retained baseline grid.
There were no numerical failures in the final training, validation or test
collections. Every correctness-passing candidate has a distinct generated
source within its workload. Source/compiler fingerprints were verified again
before the final winner remeasurement. The process monitor observed no foreign
GPU activity in 11,597 collection samples or 75 final-validation samples.

The [machine-readable summary](../experiments/results/expanded-pilot-v2-20260914/pilot-summary.json)
links the result tables to the frozen manifests, models, rankings, outcomes and
validation files. The [SVG plot](../experiments/results/expanded-pilot-v2-20260914/oracle-at-k.svg)
is also available for export.

## Frozen full-pool results

| Kernel | Generated | Pool | Compiled | Distinct compiled | Correct | Distinct correct |
| --- | --- | --- | --- | --- | --- | --- |
| GEMM | 3108 | 2060 | 2060 | 2060 | 1904 | 1904 |
| Causal attention | 1854 | 1044 | 1017 | 1017 | 737 | 737 |
| Chunk-output KDA | 3876 | 3336 | 3336 | 3336 | 3260 | 3260 |
| Softmax | 726 | 409 | 409 | 409 | 409 | 409 |

Generated counts include the retained prefix and combinations later rejected or recorded as aliases. Compiled identity is generated device-source SHA-256; binary equivalence is not claimed.

| Kernel | Current winner index | Full winner index | Current best µs | Full best µs | Current/full latency | TileTune µs | XGBoost µs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMM | 94 | 94 | 31.98 | 31.98 | 1.000× | 39.28 | 37.37 |
| Causal attention | 8 | 8 | 23.94 | 23.94 | 1.000× | 23.94 | 60.98 |
| Chunk-output KDA | 197 | 811 | 16.89 | 16.51 | 1.023× | 21.14 | 19.02 |
| Softmax | 1 | 146 | 24.28 | 20.88 | 1.163× | 24.28 | 22.90 |

These are medians from seven shuffled remeasurements including the retained-current-grid winner. Identical configurations share measurements. Winner indices were frozen using the exhaustive or online tables before this validation.

| Kernel | Method | Oracle@1 | Oracle@20 | Remeasured performance | Correct scored/total | Unique eligible scores | Correct/attempted trials | Oracle winner eligible rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GEMM | TileTune | 54.49% | 80.37% | 81.41% | 1718/1904 | 364 | 20/20 | 312 |
| GEMM | XGBoost | 77.45% | 84.71% | 85.59% | 1904/1904 | 2059 | 20/20 | 131 |
| Causal attention | TileTune | 95.21% | 100.00% | 100.00% | 324/737 | 76 | 20/20 | 2 |
| Causal attention | XGBoost | 39.98% | 39.98% | 39.26% | 737/737 | 1044 | 16/20 | 787 |
| Chunk-output KDA | TileTune | 78.76% | 78.76% | 78.07% | 20/3260 | 15 | 20/20 | unscored |
| Chunk-output KDA | XGBoost | 84.17% | 89.36% | 86.78% | 3260/3260 | 3333 | 20/20 | 123 |
| Softmax | TileTune | 60.32% | 86.81% | 85.98% | 5/409 | 5 | 5/5 | unscored |
| Softmax | XGBoost | 79.44% | 92.27% | 91.17% | 409/409 | 409 | 20/20 | 69 |

Oracle@K uses the frozen ranking and the independent exhaustive timing table. Remeasured performance is oracle median / selected-winner median. Unscored means the oracle winner was not eligible for selection. Both methods have a maximum budget of 20 trials; TileTune used only five for softmax because the other candidates were ineligible.

| Kernel | Random-20 mean | No-success fraction at K=20 |
| --- | --- | --- |
| GEMM | 85.69% | 0.00% |
| Causal attention | 46.86% | 0.00% |
| Chunk-output KDA | 93.19% | 0.00% |
| Softmax | 94.84% | 0.00% |

Random results are 500 retrospective draws per K, conditional on at least one successful configuration. They are not separately timed online tuners. The K=1/5/10/20/50 curves do not represent additional online trials.

![Frozen-ranking oracle performance by candidate budget](../experiments/results/expanded-pilot-v2-20260914/oracle-at-k.png)

## Tuning costs

| Kernel | Method | Selection s | Compilation/check/timing s | Online total s |
| --- | --- | --- | --- | --- |
| GEMM | TileTune | 82.58 | 20.74 | 103.33 |
| GEMM | XGBoost | 0.94 | 19.97 | 20.91 |
| Causal attention | TileTune | 233.40 | 69.04 | 302.44 |
| Causal attention | XGBoost | 0.59 | 71.97 | 72.56 |
| Chunk-output KDA | TileTune | 295.67 | 44.64 | 340.31 |
| Chunk-output KDA | XGBoost | 1.26 | 22.57 | 23.83 |
| Softmax | TileTune | 256.92 | 7.24 | 264.16 |
| Softmax | XGBoost | 0.48 | 16.79 | 17.27 |

These are elapsed worker tuning costs. Concurrent compile-worker durations are not summed as elapsed time. Process startup/input preparation are separately recorded in worker_wall_seconds. The independent oracle and post-selection winner validation are evaluation costs, not training labels or free online preparation.

| Kernel | Exhaustive oracle minutes |
| --- | --- |
| GEMM | 21.50 |
| Causal attention | 72.88 |
| Chunk-output KDA | 50.43 |
| Softmax | 5.15 |

TileTune primitive preparation costs 195.70 s once per device. XGBoost collection plus fitting costs 2751.76 s across the four models. The table below illustrates amortization over N comparable batches, each containing one request per family, assuming the observed online costs and reusable profile/models. It is arithmetic, not an additional experiment; it assumes fresh online tuning rather than cached winners.

| Batches N | TileTune amortized s per four-request batch | XGBoost amortized s per four-request batch |
| --- | --- | --- |
| 1 | 1205.95 | 2886.33 |
| 10 | 1029.81 | 409.74 |
| 100 | 1012.20 | 162.09 |


## Findings and follow-up work

The expanded spaces change real schedules, but more choices do not guarantee
a faster optimum. GEMM and attention retain exactly the same current-grid
winners. Softmax improves from 24.28 to 20.88 us, a 1.163× speedup. Its new winner
uses four rows, 512 columns, 256 threads, vector width four and four row groups.
KDA's new winner uses M/K/V/S tiles 16/32/128/64, two stages for the
state-contribution GEMM, zero intra-chunk stages and 256 threads. Its gain is modest: 1.023× from the
medians, or 1.011× using the median of paired round speedups. The original
exhaustive-table difference was only 0.15%; the repeated per-config timing
ranges span roughly 3%. This supports a small improvement on this case rather
than a broad KDA performance claim.

TileTune finds attention's optimum, but its coverage falls sharply for tiled
KDA and streamed softmax. It scores only 20/3,260 and 5/409 correct candidates
respectively. All 3,312 tiled KDA configs report unsupported nested serial loops,
branch-dependent schedules or unresolved memory tile sizes; the 403 added
softmax schedules also report unsupported nested serial scheduling. Attention's
unknown reasons are dominated by unresolved/inter-warp reductions. The
correct oracle winners remain in the common pool even when TileTune cannot
score them. Increasing K alone cannot overcome these coverage limits.

GEMM also demonstrates ranking error within the supported set: its oracle
winner is eligible but ranked 312th, and the 1,718 eligible configs produce only
364 distinct scores. TileTune pressure-rejects 186 correct GEMM configs. A
2026-09-15 audit joined ranking tiers to individual correctness outcomes:
the 102 unknown configs all failed shared-memory limits, and the 240
pressure-rejected configs comprise 186 correct and 54 resource-failing configs.
This corrects the earlier breakdown without changing coverage or timing results.
Its current copy-service estimate for the optimal tile
is much higher than its measured time. Intra-kernel memory reuse and the model's
sharing of bandwidth between resident CTAs are concrete directions to examine;
this run does not establish them as the cause.

XGBoost scores every declared candidate, but coverage is not ranking quality.
Its Oracle@20 is below the retrospective random-20 mean in all four cases for
this seed. It ranks the attention optimum 787th and spends four online trials on
compilation failures. The missing attention baseline training labels described
above are an observable coverage gap, not proof that they alone explain the
miss. At K=50 its frozen rankings improve considerably for KDA and softmax,
while attention remains at about 40% of oracle performance. These curves are
retrospective diagnostics; the online baseline still used at most 20 trials.

The [expanded-space repair plan](tiletune_expanded_repair_plan.md) adds the
2026-09-15 source audit, CPU profiles, corrected GEMM exclusion breakdown, and
ordered implementation milestones with validation gates.

The next changes should be evaluated separately against this frozen baseline:

1. Extend TileTune's loop-bound, memory-footprint and scheduling analysis for
   tiled KDA and streamed reductions. Add support for the relevant inter-warp
   collectives, and measure both correct-candidate coverage and winner coverage.
2. Diagnose GEMM's memory-service estimates and tied scores using held-out
   square, tall and wide shapes. Do not tune constants to the single winner here.
3. Profile and reduce CPU selection work with complete source/IR/target/profile
   cache identities. Selection alone costs 83–296 seconds here, including
   257 seconds for only 409 softmax candidates, versus 0.48–1.26 seconds for XGBoost.
4. Compare uniform and implementation-stratified XGBoost sampling at the same
   attempted-label budget, with failures still consuming the budget. Repeat
   across sampling seeds and independently varied interpolation/extrapolation
   shapes before choosing a new default. Revisit the extra conversions and
   masked work in the added attention implementation if optimizing its latency.

These are A100 results for one sampling seed and one held-out shape per family.
The other kernel families have targeted correctness/census coverage, not a full
performance comparison. This is not an end-to-end reproduction of WaveTuner.

## Running the current spaces

The configurations and measurements above are historical. The old supplementary
FP8/vector families, alternate presets, and expanded-Ampere manifest have been
removed. Current commands and the single expanded pools are documented in the
[experiment README](../experiments/README.md) and
[shared runner reference](../experiments/common/README.md).
