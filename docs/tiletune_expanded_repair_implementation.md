# Expanded TileTune repair implementation

Historical design checkpoint. The repair-study runner and its manifests are now
retired; use the [current experiments](../experiments/README.md) for executable
commands. Original measurement artifacts retain their recorded provenance.

This implements the expanded-space repair in analysis version 21. `pipeline_time`
remains the default. Kernel implementations and native compiler libraries are
unchanged. The original expanded pilot is preserved as development evidence.
The new held-out shapes were frozen before changing scoring in
`repair_heldout_ampere.json` (retired manifest).

## Changes by review stage

1. **Scheduling:** `region_schedule.py` represents ordered operations, serial
   loops, and independent software pipelines. It compresses uniform loop and CTA
   domains, splits bounded causal prefixes and tails, and drains each pipeline
   before its successor. The compiler planner is matched to each original loop
   variable. Phase dependencies and live buffers remain visible. Rectangular
   scalar guards count active coordinates; arbitrary data-dependent control flow
   remains unknown. The original single-loop timing path is retained.
2. **Collectives:** explicit fragment mappings use the existing local/shuffle/
   shared butterfly model after ownership and compiler layout validation. The
   implementation verifies full thread domains and scalar batches; unsupported
   groups, partial fragment reductions, and batched inter-warp collectives stay
   unknown. Shared traffic, barriers, workspace fences, and a reusable workspace
   are charged. Shared buffer versions follow verified compiler def/use stages
   and instruction order, so within-stage temporaries are not multiplied by
   pipeline depth. Native pipeline injection checks cover depths 1, 2, and 3.
3. **Resources and exploration:** resource violations, user-policy rejections,
   and uncertain allocation have separate diagnostic categories. With spilling
   permitted, logical register demand above 255 no longer means physical
   illegality. Such candidates retain unknown scores when spill service is
   unresolved. Explicit register/no-spill policies remain enforced.
   `TileTune + exploration` reserves 20% of K for permitted unknown-cost
   candidates, then fills unused ranked slots from that pool. Configuration
   hashes and implementation/schedule strata determine exploration. All selected
   failures consume attempts and receive no replacements.
4. **CPU work:** verified explicit mappings avoid redundant whole-graph layout
   inference. Regular scalar ownership uses exact separable bit-field counting,
   checked against a bounded reference enumerator. IR keys use structural
   equality, including collision checks, instead of diagnostic printing. A
   bounded process-local preparation cache includes the full function, target,
   pass settings, and analysis version. Cached layouts are attached to each new
   collector's buffer identities; mutable reports are copied. Profile rates do
   not affect these structural facts. Domain counting avoids rebuilding bound
   analyzers for already-constant expressions.
5. **GEMM diagnosis:** seven controlled shapes vary footprint, aspect ratio, K,
   stages, and CTA concurrency. The diagnostic subset includes 256- and
   512-register logical demand. Compiler counters are collected and charged only
   in these diagnostic runs. They are never available to selection. The known
   ranking inversion repeats across several shapes; the evidence does not yet
   separate memory reuse, transaction efficiency, and instruction issue costs
   enough to justify replacing the service formula. No winner-specific constants
   were added. Substituting the independently measured cached-memory service
   rate preserved the same five pairwise ranking inversions. A full-pool replay,
   however, improved development GEMM Oracle@20 from 80.4% to 96.3%; checking only
   that pair missed the shortlist improvement. The exact winner moved from rank
   312 to 270. The workload's 9.375 MiB input/output footprint fits the measured
   40 MiB L2, and the benchmark reuses inputs across warmups. This supports an
   explicit cached-profile comparison for that invocation pattern, while the
   frozen primary study retains its declared streaming profile. An additional
   offline check substituted the already-collected compiler register counts
   into the residency calculation. It also preserved all five inversions under
   both memory regimes. Shared-memory estimates and register allocation
   granularity remain limitations of that diagnostic; those counters never
   enter production selection.
6. **XGBoost and comparison:** uniform sampling remains the default.
   `implementation_stratified_config_hash_v1` allocates up to three initial
   attempts per implementation, round-robin if necessary, then uses integer
   largest-remainder allocation over remaining capacity. Each workload gets
   exactly `ceil(0.1 * pool_size)` attempted configurations in both training and
   validation. Failed samples are not replaced. The existing 600 rounds, depth
   10, learning rate 0.05, subsample 0.8, and patience 20 are retained.

Sampling policies are versioned independently of the unchanged feature schema.
Collection metadata must match the requested policy and seed. Old artifacts are
not rewritten to match a new execution identity. The primitive profile schema
remains unchanged because this repair uses existing measured primitives.

## Use

Pure TileTune needs no interface change. Exploration is explicit:

```python
TileTuneConfig(
    enabled=True,
    top_k=20,
    exploration_fraction=0.2,
    exploration_seed=123,
    max_spill_bytes=None,
    max_local_bytes=None,
    performance_model=profile,
)
```

The spill settings above permit exploration of allocation uncertainty. Omit
those settings to retain the default strict no-spill/local-memory policy.

For standalone XGBoost training:

```bash
python -m experiments.xgboost train \
  --train-runs TRAIN_RUNS --validation-runs VALIDATION_RUNS \
  --sampling-policy implementation_stratified_config_hash_v1 \
  --sample-fraction 0.1 --seed 123 --output MODEL.json
```

Run the predeclared A100 comparison with exact held-out tail shapes, three seeds,
K=20, both XGBoost policies, exploration, random selection, a shared exhaustive
oracle, and seven shuffled winner rounds:

```bash
CUDA_HOME=/root/cuda-12.4 CXX=/usr/bin/g++-10 \
LD_LIBRARY_PATH=/root/cuda-12.4/lib64 \
# Retired historical command: python -m experiments.common.repair_study \
  --output experiments/results/expanded-repair-study --workers 16
```

Use `--plan` to inspect the frozen protocol and `--resume` to continue it.
`repair_study` completes every seed's model and selection phase before collecting
any held-out exhaustive oracle. One primitive profile and a fixed input seed are
shared across methods and seeds. Training/validation collections remain separate
for each sampling policy and seed, so preparation costs cannot hide replaced
failures or retrospective subsampling. GPU observations are recorded by the
comparison runner, which waits for other compute processes to finish.

The training GEMM dimensions bracket the first held-out GEMM on each axis:
640×640×512 and 1024×2048×1536. “Interpolation” refers to these per-axis ranges;
“extrapolation” includes at least one size outside its training range. The
held-out manifest remains unchanged.

## Verification evidence

Implementation evidence is under
`experiments/results/expanded-repair-implementation-20260915/`.

- KDA config 811 and softmax config 146 now receive finite scores.
- Full original-pilot correct-configuration coverage is 3,260/3,260 KDA (100%),
  1,718/1,904 GEMM (90.2%), 617/737 attention (83.7%), and 313/409 softmax
  (76.5%). All 6,849 configurations were analyzed without an exception. The 120
  remaining correct attention configurations have register allocation
  uncertainty. Softmax has 85 such configurations and 11 additional unresolved
  residency estimates. These are still unknown scores.
- All 186 correct GEMM configurations previously pressure-rejected are now
  classified as allocation uncertainty when spilling is permitted. None is
  silently made finite or selected by pure TileTune.
- The broad Python run passed 547 tests and skipped 13. Its one failure was the
  existing Blackwell `sm_100a` cross-compile probe: installed CUDA 12.4 cannot
  target that architecture. After the final changes, 89 focused tests passed; eight region tests also
  passed, including two independent native pipelines in one kernel;
  the shared-memory versioning group passed 44 with five GPU skips.
- Independent small enumerations cover two sequential passes, causal prefixes,
  partial row/column tiles, guarded stores, and pipeline tail transitions.
- Exact ownership counting is checked against the reference enumerator.
- Serial and independently pipelined KDA cases reuse cached compiler plans for
  freshly constructed equivalent IR, preserving scores, work counts, resource
  decisions, and diagnostics (two tests). These KDA cases do not use the
  reduction-layout inference path; full native layout comparisons cover softmax
  and mixed MMA/reduction attention cases.
- The final ownership/cache group passes all 15 tests, including fresh-buffer
  rebinding for cached explicit layouts.
- Exploration and sampling tests cover exact budgets, deterministic ordering,
  policy exclusion, and failure accounting.
- Three coordinator tests verify that every seed freezes before shared oracle
  collection, changed selections are rejected on resume, and invalid seed lists
  are rejected.
- The full 409-configuration softmax performance comparison preserved every
  score, classification, and unknown reason. A second comparison ran the complete
  selection API, including ranking and JSON report writing, in fresh processes
  pinned to CPU 127. It decreased from 277.4 to 128.6 seconds (2.16×), with exactly
  equal rankings, resource decisions, and selected indices. Fresh serial preparation decreased
  from 236.2 to 112.4 seconds (2.10×), including IR construction and analysis.
  This compares the new scheduling implementation before/after CPU optimization;
  it does not treat cache population as free preparation.
- All 77 controlled GEMM diagnostic candidates passed correctness. The inspected
  256/512-demand candidates compiled with 255 physical registers and nonzero
  spills. Their legality does not establish a spill-free analytical timing cost.

A development-only selection check uses the preserved pilot timings. Pure
TileTune reaches Oracle@20 of 80.4% GEMM, 100% attention, 95.8% KDA, and 94.7%
softmax. The original winner ranks are respectively 312, 2, 1,749, and 36;
scoreability does not imply exact-winner ranking. Reserving four exploration
attempts reduced the pilot GEMM result to 76.8% for all three declared seeds and
left the other families unchanged. These are historical development timings,
not the held-out seven-round results, and exploration remains disabled by default.

The proposed 90% coverage, 95% median Oracle@20, and 2× CPU targets are engineering
release gates, not assumed outcomes. In particular, remaining allocation
uncertainty and the GEMM ranking inversion must be reported separately from the
scheduling and collective coverage repairs. Exploration success is not counted
as finite-score coverage.

The full three-seed study is running under
`experiments/results/expanded-repair-study-20260915/`. The earlier two-family
smoke comparison completed every method and all seven winner rounds. A second
coordinator smoke run deliberately remained under its original provenance while
the shared-memory repair landed; its oracle phase correctly refused the changed
source identity. No artifact was relabeled to bypass that check.

`gemm-full-pool-memory-sensitivity.json` records the additional development-only
memory-profile comparison. Replaying every archived streaming score with the
current scheduler gave exact equality before substituting the cached primitive
rate. The existing `experiments.common.run --memory-regime cached` option and
`load_device_profile(..., memory_regime="cached")` expose that service profile;
`pipeline_time` remains the metric. Cache state is an invocation assumption:
capacity alone does not establish reuse for arbitrary application calls. The
96.3% result uses historical development timings, not the new held-out study.

A separate GEMM-only cached-profile comparison was declared and its two held-out
shortlists frozen before any held-out exhaustive measurements. It lives under
`experiments/results/expanded-repair-gemm-cached-20260915/`. Both workloads fit
the nominal L2 capacity. The freeze used 14.35 seconds with 24 CPU workers and no
GPU measurements. Its queued execution uses the unchanged public runner, three
seeds, 20 attempted trials per case, and seven shuffled winner rounds together
with the corresponding primary-study winners. It verifies exact agreement with
the frozen rankings and records preparation, online, and remeasurement costs.
This additional comparison does not change the primary streaming protocol.

The full study resumed after an interruption between completed collection cases;
completed candidate tables were retained. Seed 123's uniform KDA validation case
has its full candidate outcomes and tuning duration, but its outer
`worker_wall_seconds` stamp was not written before interruption. Its missing
outer duration must remain explicit in preparation-cost reporting. The separate
`study-accounting-audit.json` artifact verifies deterministic sample membership,
exact attempted budgets, original candidate indices, and failure accounting.
For seed 123's smaller training cases, stratification changes three attention,
two KDA, and three softmax sample members; the softmax sample gains three
original-implementation candidates that uniform sampling missed. GEMM has one
implementation stratum, so its sampled membership is identical. Timing and
model differences between those GEMM policy runs cannot be attributed to
stratification; collections are independent repetitions with separately charged
costs.
