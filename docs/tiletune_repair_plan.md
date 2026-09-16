# TileTune defects and repair plan

2026-09-14. Reviewed analysis version 20 at base commit `bea5d199` against the
saved A100 80GB PCIe experiment. This is a proposed implementation and validation
plan; the performance figures are existing measurements, not new runs.

For the expanded configuration pools and the 2026-09-15 follow-up audit, see the
[expanded-space repair plan](tiletune_expanded_repair_plan.md). The results below
remain a record of the earlier, smaller-pool experiment.

The accompanying default change makes `pipeline_time` the ranking metric for
both `TileTuneConfig` and the portable runner. Timing still needs an explicit
matching profile and a supported schedule. `traffic_waves` remains an explicit
option, and `top_k=None` still means exhaustive tuning. Default spill/resource
policies have not been changed to the experiment's report-only settings.

## XGBoost sampling change and the historical baseline

New training defaults to a seeded 10% subset of each training and validation
configuration pool. The comparison coordinator selects before collection and
only measures those configurations. Standalone fitting also samples supplied
logs, while retaining their original collection costs. Configure this with
`compare.py --xgb-sample-fraction` or `experiments.xgboost train --sample-fraction`.
The original pools and selections are saved, failed samples are not replaced,
and test shapes and grids remain separate.

The historical v20 comparison exhaustively collected every supplied
configuration for each training and validation workload. Its models used all
successful, correctness-checked training measurements. The table below describes
that old protocol, not the new sampled default.

| Split | Supported workloads | Attempted configurations | Correct measurements | Compile failures | Use |
| --- | ---: | ---: | ---: | ---: | --- |
| Training, scales 0.25 and 0.5 | 34 | 2,160 | 1,998 | 162 | Fit trees |
| Validation, scale 0.75 | 17 | 1,080 | 999 | 81 | Early stopping |
| Fresh v20 test, scales 1.25 and 1.5 | 34 | 2,160 | 1,998 | 162 | Evaluate frozen selections |

For example, GEMM contributes 16 training workloads × 108 configurations = 1,728
training labels, plus 8 × 108 = 864 validation labels. Attention attempts 54
configurations per workload, but only 27 compile successfully, giving 162
training labels across six workloads and 81 validation labels across three.
The supplied configuration pools are identical across these splits; the
mathematical workloads differ. This evaluates new shapes using known
configuration choices, not unseen configuration values or unseen implementations.

Repeated candidate measurements are aggregated by median. Training workloads
receive equal total weight. The objective is squared error on log latency;
validation controls early stopping, and inference rejects training/validation
workload identities. The model does not consume TileTune predictions or compiler
resource counters. Training plus validation collection and fitting cost about
2,591 seconds, compared with 196 seconds of TileTune primitive profiling.

Sources: [collection coordinator](../experiments/common/comparison.py),
[sample reader](../experiments/xgboost/data.py),
[training and inference](../experiments/xgboost/model.py), and
[saved model audit](../experiments/results/ampere-pipeline-v20/xgboost-fit-audit.json).

## Confirmed defects and limitations

| Priority | Finding | Evidence and consequence |
| --- | --- | --- |
| P0 | Estimated register excess removes candidates from the search | The soft demand gate excludes 34 valid candidates, including five exhaustive winners. Attention estimates of 259 registers and an RMSNorm estimate of 257 exceed a 255-register limit without proving physical overflow. `report_only` does not restore their top-K eligibility. |
| P1 | Occupancy and spill costs are inaccurate | Across 228 selected kernels with counters, median compiler/model register usage is 1.523×. Substituting compiler registers lowers the modeled resident-CTA bound in 135 cases. Seventeen have spill/local storage whose traffic is absent from timing. These observations cover selected candidates, not the entire grid. |
| P1 | Logical memory traffic is not physical memory service | The experiment applies streaming rates to repeated CTA loads without predicting cache reuse. For one 2,560³ GEMM configuration, modeled input service alone is 1.26 ms, while the whole kernel measures 0.610 ms. GEMM predicted/measured latency has a 2.80× median bias. Cache accounting is a concrete suspect, not an established complete explanation. |
| P1 | Dependent scalar-load latency is missing | Recurrent external reads use the synchronous copy residual, which is zero in the saved profile. Byte service is charged, but this supplies no separate load-to-use dependency cost. The contribution to overall error has not been isolated. |
| P1 | Chunk KDA stage preference is wrong | For matched stage pairs, median predicted pipeline speedup is 1.069× versus 0.962× measured. Direction is correct in only 1/9 pairs with changes exceeding 10%. One fresh case has negative rank correlation and performs worse than the saved random-shortlist mean. |
| P1 | Analysis can cost more than exhaustive tuning | The six-entry recurrent-KDA, softmax and RMSNorm grids have tuning speedups of 0.60×, 0.91× and 0.91×. Recurrent KDA test1.25 spends 7.39 s selecting from six candidates; the complete brute-force tuning run takes 6.37 s. The runtime analyzes candidates serially. |
| P2 | Coverage and uncertainty handling remain narrow | Only finite eligible scores can enter top-K; there is no exploration allocation. Generic reductions in software pipelines, nested serial loops and branch-dependent schedules remain unsupported. Target recognition does not establish timing coverage for other GPUs. |
| P2 | Some diagnostics do not describe the executed model | The Ampere input-ready summary still uses the old synchronous expression although positive-stage scheduling uses the asynchronous probe. Some pressure assumptions still say MMA operand fragments are unmodeled. Broad exception catches in Ampere preparation can also classify unexpected failures as unsupported model coverage. |

Relevant implementation: [register policy](../tilelang/tiletune/register_pressure.py),
[liveness](../tilelang/tiletune/tile_liveness.py),
[ranking](../tilelang/tiletune/ranking.py),
[occupancy](../tilelang/tiletune/occupancy.py),
[pipeline timing](../tilelang/tiletune/pipeline.py),
[Ampere planning](../tilelang/tiletune/ampere.py), and
[runtime selection](../tilelang/tiletune/runtime.py).
Measured evidence: [v20 report](tiletune_ampere_pipeline.md) and
[candidate audit](../experiments/results/ampere-pipeline-v20/model-audit.json).

Version 20 already implements ordinary Ampere positive-stage scheduling, generic
serial reductions and corrected scalar counts. Those version-18 defects should
not be listed as still universally missing. The remaining deficiencies concern
accuracy, exclusions and the unsupported schedules specified above.

## Implementation order and acceptance criteria

1. **Preserve the baseline and prepare new evaluation workloads.** Treat all
   inspected v20 results as development evidence for the next revision. Freeze a
   new final test manifest before changing the model: additional aspect ratios,
   sizes around occupancy/cache transitions, attention head dimensions and short
   recurrences. Include irregular shapes where the adapter's correctness
   contract supports them. Keep model/profile/source hashes and split-identity
   checks. Do not tune decisions against the new final labels.

2. **Repair search exclusions and register ownership first.** Separate proven
   rejection from uncertain estimated excess. Improve live ranges, cast reuse,
   fragment ownership and replication. Keep unsupported timing explicitly
   unknown; offer a declared exploration share of K for uncertain, potentially
   valid configurations, chosen without their measured latencies. Do not merely
   increase the register allowance or assign an invented finite score. Validate
   the 34 known exclusions and five lost winners, plus counterexamples that
   really violate configured resource constraints. Acceptance: uncertain excess
   alone cannot make a valid candidate permanently unreachable, while proven
   rejection and correctness checks remain enforced.

3. **Improve occupancy, allocation and spill accounting.** Use architecture
   allocation granularity and better ownership/liveness evidence, and expose
   uncertainty in compiler scratch rather than presenting the tile proxy as
   exact physical allocation. Collect compiler observations on a declared
   diagnostic grid spanning tile size, stages and threads. Fix the saved
   131-register/256-thread example where the model predicts two resident CTAs
   despite their unrounded register requirement exceeding SM capacity. If a
   later strategy uses compiler counters to refine selection, count compilation
   in its budget and report it separately from pre-compilation ranking.
   Acceptance: recover the demonstrated occupancy boundary and reduce both
   resource-error tails and first-choice regret on development workloads.

4. **Diagnose and repair memory and schedule service.** Add independent probes
   for dependent scalar reads and relevant copy issue/concurrency patterns.
   Distinguish unique data, repeated logical loads and physical transactions;
   validate cache/transaction hypotheses with counters and controlled footprint
   sweeps. Preserve startup/steady-state/drain timing. For chunk KDA, compare
   otherwise identical stage-0/2/3 configurations while varying recurrence
   length, tile size and threads. Use generated code and synchronization/issue
   observations to identify missing work. Acceptance: improve stage direction
   and latency error without a per-workload correction factor or oracle-selected
   cache regime. Measure both median and tail absolute error, not just signed
   prediction bias.

5. **Reduce tuning overhead and make diagnostics consistent.** Profile analysis
   components before optimizing; reuse only results with matching IR, target,
   pass settings, compiler and profile identities. Investigate repeated layout
   inference and invariant work. Evaluate an explicit small-grid exhaustive
   strategy when analysis cannot repay its cost; report its full N-candidate
   budget rather than calling it a K-candidate result. Align timing summaries
   with the recurrence actually used and distinguish expected unsupported cases
   from unexpected analysis errors. Acceptance: selection and cache identity
   remain correct, and the small-grid strategy avoids the measured slowdown
   under comparable fresh-run conditions.

6. **Expand coverage after the shared model is validated.** Add pipelined
   reductions and additional loop/control-flow forms one at a time, each with
   a reference workload and explicit unsupported boundaries. Add other GPU
   instruction/storage paths with device-specific probes and exhaustive
   comparisons. Do not infer native instruction coverage from an architecture
   family name or reuse A100 capacity constants for another device.

## Evaluation required before calling the repair successful

- Report top-1, Oracle@K curves, remeasured chosen-kernel performance, finite-score
  coverage and eligible-winner coverage separately. The present 98.31%
  Oracle@K coexists with only 89.24% top-1 performance.
- Compare against brute force, random selection, XGBoost and Carver on matched
  supported cases. Include worst cases and family averages; near-flat grids
  must not conceal ineffective ordering. Count failed selections against K.
- Preserve the historical full-pool XGBoost artifacts. Evaluate the new default
  alongside separately labeled 5%, 25% and explicit 100% training-collection
  budgets with multiple seeds. Select
  subsets before reading labels, count failed collection attempts, keep shapes
  disjoint and account for validation collection as well as training. Existing
  exhaustive logs can simulate label-limited fitting, but fresh subset
  collection is needed to measure its actual wall-clock cost.
- Evaluate new configuration values in a separate held-out experiment. The
  historical full-pool training comparison tests shape extrapolation only within
  known kernel/configuration families.
- Report online tuning, preparation, startup and amortized total time
  separately. Validate winners in shuffled repeated rounds, isolate GPU load,
  and interpret differences in light of measurement spread.
- Run focused semantic/selection regressions for each change, then the relevant
  TileTune/experiment suite and the frozen final experiment. Existing results
  establish defects and regression cases; they cannot certify unseen-workload
  generalization for a model developed from them.
