# Plan for larger TileTune benchmark configuration spaces

2026-09-14. This is a proposed implementation and evaluation plan, based on the
current working tree. No kernel or grid changes are made by this document.

The initial implementation and A100 pilot are complete; see the
[implementation report](tiletune_config_space_implementation.md) for the delivered
presets, validation, results and remaining study scope.

The objective is to evaluate TileTune and the sampled XGBoost baseline on
hundreds to thousands of distinct, executable schedules per workload. Counts
must distinguish generated combinations, statically legal candidates, compiled
programs, and correct measurements. The desired sizes below are targets to
validate on representative shapes, not claims that the combinations already
compile or that every shape should have the same number of choices.

## Current implementation and intended scale

Counts below were obtained by calling `configurations()` on the current default
workloads. They include candidates that may fail compilation.

| Kernel family | Current pool per shape | Intended expanded scale | Main additions |
| --- | ---: | --- | --- |
| GEMM, transposed, batched, fused | 108 | 1,000–3,000 correct, distinct schedules on representative large shapes | Wider tile/stage ranges, warp partition, CTA traversal |
| Attention, causal attention | 54 | 500–2,000 | Wider query/KV tiles, independent QK/PV warp policies, compatible copy/layout choices |
| Recurrent KDA | 114 | 300–1,500 | More token/value tiles, unroll factors, stages and thread counts |
| Chunk-output KDA | 420 | 1,000–4,000 | Extend the existing independent row/key/value/causal tiles and two pipelines |
| Softmax, RMSNorm, row sum | 6 each | 100–500 each initially | Explicit row/thread layout, vector width, streamed-column implementation |
| Elementwise | 6 | 200–1,000 | Independent column tiling, vector width, thread layout |

The new KDA implementations are already present. Recurrent KDA retains six
baseline candidates and appends 108 tiled candidates. Chunk output retains 24
baseline candidates and appends 396 tiled candidates after its MMA tile/thread
constraint. The previous XGBoost comparison used the older 24-candidate chunk
pool; it does not evaluate the current 420-candidate implementation.

Row kernels have fewer independent scheduling decisions. Start with hundreds;
increase further only when another implemented schedule creates distinct work.
Input size, dtype, causal mode, epilogue and KDA chunk size remain workload
definitions, and do not count as extra configurations for the same workload.

## 1. Represent and audit configuration spaces

Extend `experiments/portable/spec.py` with named, deterministic presets:

- `current`: the exact current grids, including the new KDA candidates.
- `expanded`: wider existing axes plus implemented scheduling parameters.
- `large`: additional compatible layout and scheduling families.

Preserve the existing candidates and their order when appending choices. Store
a stable config hash as well as each frozen manifest's original index. Record
the preset version, complete generated pool, target, source hashes and rejection
reasons. Larger presets should contain the smaller preset's schedules; alias
records can preserve identity when two entries resolve to the same schedule.

Use conditional domains rather than multiplying every axis indiscriminately.
For example, a vector width must agree with the actual memory layout; an
implementation-specific knob must not silently do nothing on another
implementation; a recurrent unroll factor must divide its token tile.

Separate shared legality checks from each tuner's predictions. Exclude only
demonstrated semantic/backend constraints, such as an impossible MMA partition
or unsupported instruction family. Use a shared-memory bound for exclusion only
when the allocation is established. Uncertain register estimates, low predicted
occupancy and TileTune's inability to score a schedule must not remove that
schedule from the common benchmark pool.

Canonicalize declared aliases and inactive choices using schedule semantics.
Check representative parameter changes in lowered IR/generated code. During
the development compile census, also record normalized program identities to
detect additional duplicates; use these findings to define deterministic alias
rules before freezing final manifests. Do not give either tuner uncharged
compilation during final selection. Report all four counts: generated,
statically eligible, distinct compiled, and correctness-passing candidates.

## 2. Extend each kernel

### GEMM

First expand the existing builder's axes:

```text
block_m: 16, 32, 64, 128, 256
block_n: 16, 32, 64, 128, 256
block_k: 16, 32, 64, 128
stages:   0, 1, 2, 3, 4
threads:  128, 256
```

This is 1,000 raw combinations before dtype, layout and resource constraints.
Then expose `warp_policy = square / full_row / full_col` through `T.gemm`, and
CTA traversal with `swizzle_panel = disabled / 4 / 8` through `T.use_swizzle`.
That gives a 9,000-combination raw envelope before legality and alias handling;
the usable count must be measured. Different policy names that resolve to the
same partition do not establish three distinct schedules.

Validate transpose-dependent alignment and both input-copy orientations.
Extend boundary masks to bias/activation epilogues before admitting irregular
M/N shapes. Keep FP32 accumulation and the existing output contract. Check
FP16/BF16 first on A100; FP8 and other instruction families need target-specific
domains and validation.

Split-K and persistent GEMM are later implementation variants. Split-K requires
scratch/reduction handling and correct placement of the epilogue, with every
launch included in measured latency. Neither is a scalar flag to add to the
current builder merely to enlarge its grid.

### Attention

The current builder delegates to `example_mha_fwd_bshd.py`, which fixes both GEMM
warp policies to `FullRow`. Introduce an explicitly parameterized implementation
with defaults that preserve the existing behavior:

```text
block_M:    16, 32, 64, 128
block_N:    16, 32, 64, 128, 256
num_stages: 0, 1, 2, 3, 4
threads:    128, 256
qk_policy:  square, full_row, full_col
pv_policy:  square, full_row, full_col
```

The tile/stage/thread grid has 200 raw combinations; two policy axes give a
1,800-combination envelope. QK, softmax and PV share fragments: admit only policy
pairs for which a compatible layout is implemented, including any explicit
conversion and its cost. Next add tested copy widths of 1, 2, 4 and 8 elements
and alternative shared-memory layouts where they produce different code.
Choose these conditionally rather than blindly multiplying the entire pool.

Retain stable online softmax, causal masking and output dtype. Validate partial
query/KV tiles and head dimensions separately. Split-KV requires a numerically
correct softmax-state merge and a multi-kernel measurement contract, so defer
it until the single-kernel variants are established.

### KDA

Build on the existing tiled implementations and their tests.

For recurrent KDA, explore:

```text
block_v: 8, 16, 32, 64, 128
block_t: 1, 2, 4, 8, 16, 32, 64
stages:  0, 1, 2, 3, 4
unroll:  1, 2, 4, 8, restricted to divisors of block_t
threads: 64, 128, 256, subject to backend support
```

There are 1,650 tiled combinations after the unroll divisibility rule, before
other constraints and alias removal. Preserve ordered recurrence, gate
conventions, intermediate casts, tail masking and the final FP32 state.

For chunk output, start with:

```text
block_m:      16, 32, 64, 128
block_k:      16, 32, 64
block_v:      32, 64, 128
block_s:      16, 32, 64, 128
stages:       0, 1, 2, 3
intra_stages: 0, 1, 2
threads:      128, 256
```

This is 3,456 raw tiled combinations. Check both GEMMs' partitions, intermediate
storage and causal bounds. Add key tiles of 128, deeper pipelines and independent
GEMM policies to `large` only after validating the initial space. Keep
`chunk_size` fixed within a workload and preserve chunk boundaries.

Use additional workloads with longer sequences and larger key/value/chunk
dimensions. A one-iteration reduction may not exercise different pipeline
depths; increasing the number of stage labels on that shape is not evidence of
a richer effective search space.

### Softmax, RMSNorm and row reduction

The current `row_case` allocates a whole power-of-two padded row in fragments.
Add explicit row/column ownership and copy vectorization, starting from:

```text
block_rows: 1, 2, 4, 8, 16
threads:    64, 128, 256, 512
vector:     1, 2, 4, 8 elements
layout:     three implemented row/column ownership patterns
```

This is a 240-combination design envelope, not 240 guaranteed distinct programs.
The layouts need actual fragment mappings and compatible reduction ownership.
Thread counts and vector widths remain conditional on shape/backend support.

Add a streamed-column implementation with `block_cols` in
`128, 256, 512, 1024, 2048`. Softmax must compute a stable complete-row
normalizer before producing outputs; RMSNorm must finish the row's sum of
squares; row sum can accumulate column tiles directly. Include rereads and all
passes in timing. Keep these domains conditional on implementation so whole-row
variants do not acquire a meaningless column-block parameter.

### Elementwise

Separate elementwise scheduling from the reduction-oriented `row_case`.
Expose independent tile columns and vector width:

```text
block_rows: 1, 2, 4, 8
block_cols: 64, 128, 256, 512, 1024
threads:    64, 128, 256, 512
vector:     1, 2, 4, 8 elements
```

This provides 320 raw combinations, with additional distinct ownership layouts
available for a larger preset. Preserve the exact elementwise expression and
mask both row and column tails.

## 3. Make both tuners ready for the new axes

Keep XGBoost's baseline settings: at most 600 rounds, depth 10, learning rate
0.05, row subsampling 0.8, and early stopping patience 20. Keep the seeded 10%
configuration subset for each training and validation workload; choose it
before measurements, and do not replace failed samples. For a 3,000-candidate
pool this means 300 attempted labels per training shape, not all 3,000.

The existing feature extractor already accepts numeric and categorical knobs.
However, its schema is currently inferred only from successful sampled rows.
An implementation-specific feature absent from those rows can therefore cause
inference to reject other candidates. Define the schema from the declared,
unlabeled search-space specification, retain missing/unknown encoding, version
the artifact schema, and fit only on the sampled labels. Recollect and retrain
models after kernel changes; keep source/domain mismatch checks.

Use `pipeline_time` for TileTune. Audit how each new parameter affects its IR
analysis: tile/stage changes, warp ownership, copy width, traversal and layout
may have different levels of support. Report ties and unsupported schedules
explicitly. Keep every correct candidate in the independent exhaustive oracle,
including candidates TileTune cannot score or makes ineligible. Do not invent
scores or narrow the common pool to hide missing model coverage.

Profile analysis cost as the pool grows. Cache invariant work only with complete
IR/target/compiler/pass/profile identities. Optimize selection overhead as a
separate, validated change so the expanded-space comparison remains traceable.

## 4. Validate and compare in stages

1. **Inventory and infrastructure:** freeze the current grids, implement named
   spaces and legality/alias reports, and handle the XGBoost schema. Add focused
   tests for real contracts: old identities, invalid combinations, new axes at
   inference and unchanged sample budgets.
2. **GEMM pilot:** implement the wider axes and new policy/traversal parameters;
   compile and check a development census spanning tiles, stages, layouts,
   transpose modes, tails and FP16/BF16. Run the first larger-pool comparison.
3. **Attention and KDA:** introduce compatible attention layouts, extend the
   existing KDA domains, and validate attention masking plus recurrent state
   and chunk-boundary behavior. Treat unsupported model coverage as a measured
   result, and address it in subsequent model revisions.
4. **Row and elementwise implementations:** validate ownership and multi-pass
   reductions before expanding these pools. Use small/large and irregular row
   widths to exercise the new scheduling decisions.
5. **Final experiment:** freeze source, manifests, training/validation/test
   workload splits and budgets after development. Start on A100, then validate
   new target-specific spaces on available hardware. Maintain one common pool
   per workload/device for both tuners.

Use a small pilot with two training shapes, one validation shape and two test
shapes per selected family. For the full study, use at least three sampling
seeds and more independently varied shapes: square/tall/wide and varying K for
GEMM; head dimension/sequence/causal mode for attention; key/value dimensions,
sequence and chunk length for KDA; row count/width for row kernels. Freeze final
test shapes before examining their latency labels. Report interpolation and
extrapolation separately; holding out entire config values or implementations
is a separate experiment from generalization to new shapes.

The runner currently uses `min(top_k, ceil(pool_size * budget_fraction))`, with
defaults `top_k=20` and `budget_fraction=0.1`. Thus a 3,000-candidate pool still
gets only 20 online trials by default. Make this explicit, and evaluate matched
budgets such as K = 1, 5, 10, 20 and 50, alongside separately labeled fraction
budgets. Training fraction and online K must remain independent. Failed online
trials consume the same budget for both methods.

Freeze each ranking before collecting its final exhaustive oracle. Check every
measured candidate against an independent reference, retain failures with
reasons, and remeasure selected winners in shuffled rounds on an isolated GPU.
Shard/checkpoint long development/oracle collections while preserving config
identities; include compilation costs and measure concurrent workers' impact
on benchmark isolation. Bound candidate runtime and report timeouts explicitly.

Report:

- Generated, legal, distinct compiled and correct config counts per shape.
- Best latency in the old and expanded pools, showing whether added schedules
  actually improve attainable performance.
- Top-1 and Oracle@K against the expanded oracle, remeasured winner latency,
  ties, invalid selections and TileTune score/eligible-winner coverage.
- Performance against random shortlists at the same K and across seeds.
- Offline collection/fitting or primitive profiling, CPU selection,
  compilation, online measurement, and total/amortized tuning cost separately.
- Cases where the grid is almost flat, where new schedules win, and where
  the model fails to rank them.

Completion means the parameters change real schedules, the valid space reaches
the stated scale where the workload supports it, correctness holds, sampling
remains partial, and the comparison exposes both performance and coverage.
Increasing the number of dictionaries alone does not satisfy the objective.
