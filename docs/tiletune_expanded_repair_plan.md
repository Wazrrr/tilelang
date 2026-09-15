# Repair plan for TileTune on expanded configuration spaces

2026-09-15. This plan follows the completed A100 expanded-pool pilot and a new
source audit plus CPU profiling of eight representative configurations. It
proposes model and selection changes; those changes have not been implemented.
The original [repair plan](tiletune_repair_plan.md) describes earlier v20
experiments. Coverage is now the first priority because the expanded kernels
exercise structures outside the current timing model.

## Findings supported by the expanded pilot

The baseline uses `pipeline_time`, a maximum of 20 online trials, and the fixed
XGBoost settings with a seeded 10% attempted-configuration budget for both
training and validation. The four exhaustive pools contain 6,310 correct
configurations. Oracle@20 below uses the frozen exhaustive timing table;
random values are means of 500 retrospective shortlists, not separately timed
online runs.

| Family | Correct configs | TileTune scored | TileTune Oracle@20 | XGBoost Oracle@20 | Random-20 mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| GEMM | 1,904 | 1,718 | 80.37% | 84.71% | 85.69% |
| Causal attention | 737 | 324 | 100.00% | 39.98% | 46.86% |
| Chunk-output KDA | 3,260 | 20 | 78.76% | 89.36% | 93.19% |
| Softmax | 409 | 5 | 86.81% | 92.27% | 94.84% |

### 1. Multiple loops are mistaken for unsupported nested scheduling

The generic family stores a single `loop` and chooses it only when there is one
candidate loop. `analyze_pipeline` then rejects other serial loops, missing main
loops and operation predicates. Ampere pipeline planning also requires exactly
one unconditional pipeline for the entire kernel.

This directly conflicts with the implemented kernels:

- Streamed softmax has two **sequential** column loops: statistics, then output.
- Tiled chunk KDA has two sequential GEMM loops: the state contribution, then
  the intra-chunk contribution. Either loop can have its own pipeline depth.
- KDA's zero-stage causal bound depends on the CTA row; its epilogue has a
  bounds guard. These are not necessarily arbitrary data-dependent control flow.

Thus the diagnostic "nested serial loop scheduling is not modeled" also covers
sibling loops that are not nested. All 3,312 tiled KDA configs and all 403 added
softmax configs encounter structural timing restrictions. The new oracle winners,
KDA config 811 and softmax config 146, are unscored.

Code: [family selection](../tilelang/tiletune/families/__init__.py),
[pipeline analysis](../tilelang/tiletune/pipeline.py),
[Ampere planning](../tilelang/tiletune/ampere.py),
[CTA work distribution](../tilelang/tiletune/schedule.py),
[kernel implementations](../experiments/portable/kernels.py),
and [streamed reductions](../experiments/portable/kernels_expanded.py).

### 2. Explicit ownership cannot use an existing inter-warp reduction model

`fragment_reduction_work` already models supported inter-warp butterfly rounds,
shared-memory work and barriers. However, `reduction_work` passes
`allow_interwarp=inferred`. The `inferred` flag is set for the compiler-inferred
layout path, but not when the selected layout comes from an explicit annotation.
Even when an inferred layout is available too, the explicit layout is selected
first. Valid explicit layouts in the expanded kernels therefore fail this gate.

This explains a concrete part of attention/softmax coverage loss. It does not
justify enabling every explicit layout blindly: layout agreement, batch shape,
subgroup structure and the compiler's actual collective must still be checked.

Code: [reduction ownership and service](../tilelang/tiletune/compute.py).

### 3. Register-demand policy excludes executable GEMM configurations

A new join by original config index corrected the previous report's breakdown:

- All **186 correct but unscored GEMM configs are pressure-rejected**.
- Their reasons are modeled tile demand of 256 registers (141 configs) or 512
  registers (45 configs), versus the 255-register allowance.
- The other 54 pressure-rejected configs fail shared-memory limits.
- All 102 configs in the `unknown` tier also fail shared-memory limits.

The code distinguishes a modeled demand lower bound from estimated liveness,
but a logical demand above the allowance is not by itself evidence that an
executable kernel is impossible. Compiler allocation and spilling can differ
from the model. The pilot disabled explicit spill/local-byte limits. In
`report_only`, `keep` remains true, yet `rank_records` still assigns the
`pressure_rejected` tier using `would_reject`; top-K excludes that tier.

Separately, `apply_ranking_metric` suppresses scores for estimated demand above
the soft allowance. We need to distinguish resource illegality, an explicit user
policy, and uncertainty about allocation/spill cost.

Code: [register policy](../tilelang/tiletune/register_pressure.py),
[ranking and selection](../tilelang/tiletune/ranking.py).
Evidence: [per-config exclusion audit](../experiments/results/expanded-repair-audit-20260915/gemm-exclusions.json).
The earlier [implementation report](tiletune_config_space_implementation.md) has
been corrected; its total coverage and performance results are unchanged.

### 4. GEMM has a ranking error even within supported candidates

Config 94 is the exhaustive winner but ranks 312th, tied across ranks 310–321.
The model predicts 67.09 us versus 31.71 us measured in the oracle table.
Selected config 101 is predicted at 45.30 us versus 39.46 us measured there.

The model charges config 94 about 4,490 copy-service cycles per iteration for
20 iterations and two concurrent CTAs, versus 1,497 cycles for 40 iterations and
one concurrent CTA for config 101. Logical repeated loads are serviced using
the streaming rate; the memory model explicitly lacks a transaction/coalescing
or inter-CTA cache model. Within-launch reuse, transaction efficiency and
bandwidth sharing are testable explanations, not established sole causes.
L2 flushing before each timed launch does not remove reuse within that launch.
The 1,718 eligible GEMM configurations have only 364 distinct scores.

Code: [logical traffic](../tilelang/tiletune/global_memory.py),
[pipeline service](../tilelang/tiletune/pipeline.py),
[occupancy](../tilelang/tiletune/occupancy.py),
and [wave ranking](../tilelang/tiletune/ranking.py).

### 5. CPU analysis is expensive, with identified hot paths

The saved serial selection-stage timings separate IR construction from analysis:

| Family | Build IR, s | Analyze, s | Total selection, s |
| --- | ---: | ---: | ---: |
| GEMM | 43.27 | 38.60 | 82.58 |
| Attention | 92.07 | 140.72 | 233.40 |
| Chunk KDA | 112.75 | 181.17 | 295.67 |
| Softmax | 31.00 | 225.65 | 256.92 |

`prepare_top_k` processes candidates serially. Sorting/report overhead cannot
explain most of these totals. A CPU-only cProfile audit identified:

- Repeated Ampere preparation/layout inference: softmax config 146's second
  profiled analysis spent 0.466 of 0.597 seconds in `prepare_analysis`.
- Scalar ownership enumeration: softmax config 408 spent 0.707 of 1.243 seconds
  in `scalar_fragment_work`, plus 0.427 seconds in `prepare_analysis`.
- `scalar_fragment_work` materializes per-element/per-replica ownership points
  and counts projected coordinate sets separately for scalar expressions.
- Global-memory key construction invokes string representations of symbolic
  loop expressions. The first KDA profile also exposed a printer warning and
  a cold-start outlier; it should not be treated as steady-state cost.

These are instrumented CPU measurements, not replacement online timing numbers.
Both repetitions run in one process, so cache/import state differs. They locate
optimization work without proving a particular speedup.

Code: [runtime loop](../tilelang/tiletune/runtime.py),
[analysis pipeline](../tilelang/tiletune/engine.py),
[Ampere layout preparation](../tilelang/tiletune/ampere.py),
and [scalar ownership counting](../tilelang/tiletune/compute.py).
Evidence: [CPU profiles](../experiments/results/expanded-repair-audit-20260915/cpu-profile.json)
and [saved selection-stage breakdown](../experiments/results/expanded-repair-audit-20260915/selection-breakdown.json).

### 6. XGBoost's uniform sampling can omit a small implementation family

Across both attention training shapes, only three original-implementation
configs were sampled, and all three failed compilation. All 163 training labels
therefore describe the new tiled implementation. Validation has four successful
original-implementation examples, but validation controls stopping rather than
fitting tree splits. The model ranks the attention optimum 787th and spends four
online trials on compilation failures.

The feature schema already accepts every declared parameter; accepting an input
is different from having training evidence for it. This is a confirmed sampling
coverage gap. A controlled sampling-policy comparison is required to establish
its contribution to prediction error. The fact that XGBoost falls below the
random-20 mean in every case also calls for evaluation of top-K selection quality,
not just aggregate regression correlation.

Code: [sampling](../experiments/xgboost/sampling.py),
[model fitting](../experiments/xgboost/model.py).
Evidence: [attention training coverage](../experiments/results/expanded-pilot-v2-20260914/attention-training-coverage.json).

## Ordered implementation plan

### Stage 0 — Freeze evidence and define success

Keep the completed pilot as development/regression evidence. Preserve its
source archive, native identity, config IDs, profiles, models and uniform
sampling policy. Freeze a new test manifest before changing models. Capture
minimal analysis fixtures for softmax 146, KDA 811, GEMM 94/101, representative
pressure exclusions, and valid explicit inter-warp layouts.

Record reason codes independently for structural coverage, memory bounds,
collectives, allocation uncertainty, explicit policy rejection and compiler
failure. Keep the human-readable explanation and the operation/loop responsible.

### Stage 1 — Model sequential loops and bounded control flow

Replace the single-main-loop assumption with a region schedule representing
ordered operations, sequential regions, serial loops and pipelined loops.
Retain existing single-loop behavior as a compatibility case. Start with
sequential serial regions for streamed sum/RMSNorm/softmax, then KDA's two
pipelines and their shared output accumulator; general nested recurrence comes
next.

For each region, track loop domains, memory visits, state carried between
regions, live buffers, and its timing summary. Compose sequential costs in
program order. Drain a pipeline before dependent later work unless compiler
scheduling proves an overlap. Adapt Ampere plan extraction and operation mapping
per pipeline rather than requiring one plan for the whole kernel. Generalize
CTA work grouping to bounded work summaries, not just one iteration count.

Resolve affine/ceildiv/min/modulo bounds under both CTA and loop domains. Split
interior and tail classes where necessary; count masked bytes and partial tiles
without charging full work as a known exact value. Keep launch/workload-scale
loops symbolic or compressed, rather than enumerating the full grid.
Distinguish static branches, tail predicates, CTA-dependent causal prefixes and
runtime data-dependent branches. A predicated store must not invalidate unrelated
pipeline work. Arbitrary unresolved control flow remains explicitly unknown.

**Acceptance:** analytic byte/work counts match small independently enumerated
fixtures; every loop/pass is counted once; loop-carried state and pipeline
startup/drain remain correct. KDA 811 and softmax 146 become timing-supported
once their collective/memory prerequisites are met. Existing GEMM/attention
schedule regressions retain their behavior.

### Stage 2 — Unify verified fragment ownership and collective modeling

Create one ownership-fact path for explicit and inferred layouts. Permit the
existing inter-warp model when target, fragment mapping, reduction axis,
replication, compiler collective and batch constraints have been validated.
Include local/shuffle/shared work, synchronization and workspace reuse.
Add primitive probes only where existing profile rates are insufficient; version
the profile schema and reject incompatible profiles.

Do not turn the gate on unconditionally. Add cases where an explicit and an
inferred equivalent layout yield the same work, and counterexamples with
unsupported lane groups/batching that must stay unknown. Check multiple row
thread groups and partial row/column tiles.

**Acceptance:** supported explicit layouts no longer become unknown merely
because of annotation provenance; the new softmax winner and representative
attention layouts receive justified finite costs. GPU kernel semantics and
correctness are unchanged by analysis.

### Stage 3 — Repair resource classification and unused-budget handling

Represent hard physical constraints, explicit user policies, and uncertain
logical demand separately. Audit the 186 correct GEMM exclusions with compiler
allocation/spill observations on a declared diagnostic subset. Retain actual
hardware limits and explicit no-spill requirements; do not make logical demand
an automatic hardware-illegality claim. Improve ownership/liveness and allocation
rounding, and model spill service where evidence supports it. Otherwise retain
an uncertainty reason rather than inventing a precise latency.

As a separately named comparison, add `TileTune + exploration`: reserve a
predeclared part of K, initially 20%, for seeded candidates with unsupported or
uncertain costs, and fill unused ranked slots from that pool. Stratify the
exploration using declared implementations/scheduling features. Candidates that
violate proven constraints or explicit policies remain excluded. The shortlist
is fixed before its candidate measurements; failed trials consume K, with no
replacement. The original pure-ranking baseline remains reproducible.

**Acceptance:** supported-but-uncertain candidates are distinguishable from
illegal ones. The hybrid can use all 20 attempts where enough permitted
candidates exist, without reporting explored configs as analytically scored.
Tests cover deterministic budgets, ties, no leakage/replacement, genuine resource
failures, and explicit strict policies. Exploration success alone does not count
as a model-coverage repair.

### Stage 4 — Reduce CPU selection cost without changing estimates

Reuse validated explicit ownership facts to avoid unnecessary whole-module
layout inference. Cache immutable structural facts with complete expression,
layout, shape, dtype, target, pass/compiler and relevant profile identities.
Do not reuse stale buffer ObjectRefs across different PrimFuncs.

Replace repeated ownership-point enumeration with exact counting for supported
regular mappings; retain a bounded reference enumerator for validation and
unusual layouts. Reuse expression dependency projections and safe IR templates
where invariant work is demonstrated. Remove expensive diagnostic string
construction from hot identity paths. Evaluate bounded process workers only
after these serial hot paths are addressed and TVM/context isolation is tested.

**Acceptance:** a dedicated performance-only revision produces identical
scores, tiers, reasons and deterministic selections before/after optimization;
cache invalidation and non-mutation checks pass. Proposed target: at least 2×
lower fresh selection time per family at matched CPU resources, with separate
cold/warm results. Do not report cached work as free cold-start analysis.

### Stage 5 — Improve supported GEMM ranking

Use controlled tile/footprint/CTA-concurrency sweeps and primitive probes to
separate logical bytes, transaction efficiency, reuse and effective memory
service. Audit occupancy/register allocation on those diagnostic cases. Any
counter-driven refinement must charge its required compilation/profiling cost;
final pre-compilation ranking cannot receive uncharged candidate counters.

Add justified physical-service or uncertainty terms while retaining both logical
traffic and assumptions in diagnostics. Check otherwise-matched stage/warp-policy
pairs and the GEMM 94/101 inversion. Avoid workload-specific multipliers,
oracle-chosen cache regimes or fixed bonuses for known winner configurations.

**Acceptance:** top-K regret, meaningful stage-direction errors and prediction
error tails improve across independent square/tall/wide and varying-K shapes.
Explain residual ties. Recovering one inspected winner is a regression check,
not evidence of generalization.

### Stage 6 — Repair XGBoost sampling and run the comparative study

Add a versioned, deterministic implementation-stratified sampler alongside the
unchanged uniform baseline. Keep the total count exactly `ceil(0.1 * pool_size)`.
Allocate a small minimum attempted quota to each implementation when the budget
allows, then distribute the remainder proportionally with deterministic rounding.
Sample within each stratum using seed/config hashes, with no latency inputs.
Record unmet coverage and deterministic quota behavior when the budget is too
small; never enlarge it silently. Failed samples are not replaced, so an
attempted quota is not a guarantee of successful labels.

Apply the policy to training and validation separately. Keep the baseline's
600-round/depth-10/0.05/0.8 settings and patience 20 fixed for this ablation.
Update sampling-plan validation, artifact policy/version and resume identities.
Retain all-config *unlabeled* schema construction and sampled-label-only fitting.
Evaluate a failure predictor or a different ranking objective only as subsequent
ablations if sampling coverage does not explain the remaining misses.

**Acceptance:** exact sample budgets and reproducibility hold; implementation
coverage is reported per seed; valid strata do not disappear silently. Compare
three sampling seeds against uniform sampling, rather than selecting a favorable
seed or filling failed samples from oracle results.

## Final evaluation and release gates

Start with at least two new test shapes per family: independently vary GEMM
aspect ratio/K; attention sequence/head dimension/causality; KDA key/value/chunk
sizes with valid chunk boundaries; and reduction row/column sizes including tails.
Separate interpolation from extrapolation. Keep test shapes disjoint from new
training and validation workloads. Preserve identical kernel implementations,
config pools and compiler identities when comparing model revisions. Evaluate
any attention-kernel optimization as a separate kernel revision.

Freeze every method's model/ranking and all seed-specific shortlists before
collecting each final oracle. An oracle table can be shared across seeds on the
same frozen workload; it need not be recompiled three times. Use fresh partial
collection when reporting training cost. Preserve strict execution-domain checks:
run historical models in their matching frozen environment or legitimately
recollect/retrain; do not rewrite source hashes to make an old model load.

Compare pure TileTune, the separately labeled exploration variant, uniform and
stratified XGBoost, random K and exhaustive search. Carver remains unsupported
where its adapter cannot represent the expanded axes. Measure actual end-to-end
K=20 runs; label K=1/5/10/20/50 curves as retrospective unless those budgets are
also run end to end. Separate finite-score coverage, physical feasibility and
successful trials. Remeasure winners in at least seven shuffled rounds with an
isolated GPU, and report preparation, online selection/compilation/measurement,
and amortized costs separately.

Proposed engineering gates, not achieved results:

- At least 90% finite-score coverage of correct configurations in each supported
  expanded family, with the known regression winners scoreable and remaining
  unsupported structures explicitly listed.
- Target at least 95% median Oracle@20 on the new test matrix. Report each family,
  worst cases, and differences from random/XGBoost across seeds; a high pooled
  average must not hide a failing family.
- At least 2× lower fresh CPU selection time per family at matched resources;
  accuracy changes and caching effects are reported independently.
- No silent correctness regressions, budget expansion, oracle leakage or hidden
  replacement of failed trials. Kernel correctness repairs discovered during
  development require a new frozen source and new final evaluation.

The first implementation milestone should be **sequential-loop/memory support
plus verified explicit collectives**. These unblock the KDA/softmax winners.
Resource classification and measured hot-path optimization follow; GEMM service
calibration and the XGBoost sampler are separate ablations. More boosting rounds
or simply increasing K will not repair the current structural coverage gaps.
