# Simplified TileTune memory ranking on B200

The `dev-b200-tiletune` worktree adds a family-independent
`ranking_metric="memory"` path. Its B200 ordering keeps IR pipeline depth ahead
of logical request count within equal byte-waves. This keeps the model small,
retains the useful H200 request refinement, and avoids fitted compute,
occupancy, spill, and family-specific timing rules.

## Contract

```text
PrimFunc -> generic operator/access/dependency/storage facts
         -> target SM count and IR pipeline depth
         -> memory score and conservative tie ranks
```

No GEMM, attention, KDA, grouped-GEMM, FP8-GEMM, or softmax adapter participates
in the score. Primitive semantics remain common: reads, writes, reductions,
copies, and MMA regions are collected wherever they occur. Explicit GEMM or
attention specialization hints are rejected in memory mode. The old
`pipeline_time` and `traffic_waves` paths retain their existing family policies.

This makes a new PrimFunc analyzable without an accessory family function when
its global memory effects and loop extents are visible to the collector.
Unresolved global effects remain unknown instead of being guessed. The
standalone softmax unit test exercises masked global accesses, elementwise
expressions, and two reductions while assertions disable family construction,
timing, occupancy, and warp-specialization calls.

## Score and pruning rule

```text
byte_work = sum(access.bytes * access.visits)
waves = ceil(grid_blocks / SM_count)
logical_byte_waves = byte_work * waves
logical_access_waves = memory_events * waves
score = 65535 * B * (B + 1) // 2
      + (65535 - pipeline_depth) * (B + 1)
      + E
where B = logical_byte_waves and E = logical_access_waves
display key = (score, original_index)
rank = last position occupied by an equal-score group
```

Every nonempty logical access transfers at least one byte, so `0 <= E <= B`.
The summed band widths therefore encode `(B, -pipeline_depth, E)` exactly with
Python integers: one fewer byte-wave always wins; at equal bytes deeper
buffering wins; at equal bytes and depth fewer logical requests win. Original
index orders only identical triples. Default runtime top-K selection expands
through the complete boundary group and reports the excess. Strict selection
retains a group only when its conservative tail rank fits the budget.

Copying H200's exact `(B, E, -pipeline_depth)` order was tested but not kept. It
improved attention while moving three B200 FP8 oracle winners from roughly 17%
to 42% of their pools, because request count overrode the useful stage-depth
signal. The B200-adapted `(B, -pipeline_depth, E)` order improved or preserved
every oracle rank in both frozen and stored-live-fact audits.

The score counts logical, padded accesses. It is not a cache, coalescing,
transaction, bandwidth, compute, or physical-occupancy model. Dependency and
storage reports remain available for inspection and explicit resource policies,
but they do not become an implicit timing formula.

## Archived space-version-11 compile-valid pool migration check

Space version 11 removed configurations that failed B200 compilation and kept
every pool above 500 candidates. The 25 stored live reports were rescored over
the retained configurations. The attention stage-6–12 and KDA stage-8 additions
reuse the same global-access and grid facts as their stage-5/stage-7 counterparts
and change only the declared IR pipeline depth. The one added GEMM example launch
was conservatively allowed to rank ahead of each prior oracle.

| Family | Cases | Current pool | Prior oracle within 20% | Prior oracle within 50% | Worst tail-rank share |
|---|---:|---:|---:|---:|---:|
| FlashAttention | 5 | 520 | 1/5 | 5/5 | 21.92% |
| GEMM | 5 | 1,473 | 5/5 | 5/5 | 1.09% |
| FP8 GEMM | 5 | 533 | 5/5 | 5/5 | 8.44% |
| Grouped GEMM | 5 | 576 | 5/5 | 5/5 | 7.29% |
| KDA intra | 5 | 513 | 0/5 | 5/5 | 35.09% |
| **All** | **25** | — | **16/25** | **25/25** | **35.09%** |

A ceil-rounded 36% budget retains all 25 previously measured oracle winners,
so the requested 50% cutoff has margin. This is a migration check, not a new
exhaustive oracle: 337 newly admitted attention/KDA configurations and the added
GEMM example launch have compiler validation but no latency measurements, per
the instruction not to run a large sweep. A fresh full oracle is required before
calling the best configuration of the version-11 pool measured.

## Space-version-10 frozen-pool replay

The offline replay used the completed
`b200-v10-five-family-top20-20260917` study. It reconstructs score inputs from
the archived TileTune operation facts, freezes every ranking, then opens the
hash-verified exhaustive oracle only to evaluate the winner's rank. It does not
compile kernels, execute kernels, change the pools, or run GPU measurements.

| Family | Cases | Pool per case | Within 20% | Within 50% | Worst tail-rank share |
|---|---:|---:|---:|---:|---:|
| FlashAttention | 5 | 576 | 5/5 | 5/5 | 7.64% |
| GEMM | 5 | 2,304 | 5/5 | 5/5 | 0.69% |
| FP8 GEMM | 5 | 533 | 5/5 | 5/5 | 8.44% |
| Grouped GEMM | 5 | 576 | 3/5 | 5/5 | 39.24% |
| KDA | 5 | 512 | 0/5 | 5/5 | 31.25% |
| **All** | **25** | — | **18/25** | **25/25** | **39.24%** |

A uniform ceil-rounded 40% budget is sufficient for all 25 fixed-pool oracle
winners; the requested 50% guarantee has margin. The limiting case is
`grouped_gemm_prefill`, whose equal-score group ends at rank 226/576. These are
retrospective results for this sealed pool, not a general performance guarantee.

| Formula | Within 20% | Within 50% | Mean oracle share | Worst share |
|---|---:|---:|---:|---:|
| Previous B200 `(B, -D)` | 18/25 | 25/25 | 15.12% | 40.28% |
| H200 `(B, E, -D)` | 15/25 | 25/25 | 15.03% | 42.21% |
| B200 adapted `(B, -D, E)` | 18/25 | 25/25 | 12.29% | 39.24% |

That frozen replay predates direct bounded-`While` handling. Its archived
operation rows did not retain the enclosing `While` nodes, so the replay could
not prove the persistent scheduler's visit bound. The earlier 16.7% FP8 control
result and the refined 8.44% replay result both identify the correct stage-six
group, but neither is the source of truth for live selection.

## Declared metadata and analysis cost

`TileTuneConfig.input_values` supplies verified read-only integer metadata to
the analysis view without changing the compiled PrimFunc. Lean memory mode
resolves lookup indices, loop bounds, and access extents, but defers complete
address and predicate simplification. Uncertain deferred results retry the eager
path. `memory_diagnostics=True` also uses eager resolution.

A paired CPU benchmark used the five final grouped-GEMM cases and config indices
`0`, `N//2`, and `N-1`: 15 prebuilt PrimFuncs, one warmup, five alternating
measurements per path. Scores, byte/access waves, and complete resource decisions
matched. Mean per-sample median analysis time fell from **512.870 ms** eager to
**25.671 ms** deferred, a **19.98x speedup**. Generation, compilation,
serialization, and GPU execution were excluded.

## Stored live-fact rescore

The completed live run's 25 full analysis reports were also rescored without
opening their oracle guards until after ranking. Compared with the previous
formula, the adapted order keeps 20/25 winners within 20%, keeps 25/25 within
50%, reduces mean oracle share from 12.24% to 9.42%, and leaves the worst share
at 31.25%. H200's order keeps 25/25 within 50% but only 17/25 within 20% and
worsens the worst share to 42.21%. No kernels were recompiled or timed for this
rescore.

## Previous strict alpha=50% combined GPU audit

The previous GPU 0/1 combined run used `--alpha 0.5`, grouped compilation with
groups of four, compile/benchmark pipelining, two-GPU benchmarking, and
post-compile spill/local-memory detection. The strict integer budget is
`floor(pool_size * alpha)`. The brute-force side is read from the verified
fixed-pool baseline bundles and is not rerun.

Every live PrimFunc is analyzed directly. A positive-step `While` is scored
only when its initialization, limit, unconditional update, and conservative
maximum visit count are provable. Manual pipeline depth is read from an
asynchronous global-to-shared copy whose leading shared axis is indexed modulo
that axis's extent. Other `While` loops remain unknown.

| Family | Pool per case | Alpha budget | Final selected range | Worst oracle tail rank |
|---|---:|---:|---:|---:|
| GEMM | 2,304 | 1,152 | 1,136–1,152 | 24/2,304 (1.04%) |
| FlashAttention | 576 | 288 | 288 | 60/576 (10.42%) |
| KDA | 512 | 256 | 256 | 160/512 (31.25%) |
| FP8 GEMM | 533 | 266 | 200 | 89/533 (16.70%) |
| Grouped GEMM | 576 | 288 | 288 | 42/576 (7.29%) |

FP8 has a leading 89-config score group followed by 111-config groups. The
266-config budget admits the first two groups (200 configs); the next group's
tail rank is 311, so strict selection excludes it in full. All five FP8 oracle
winners have tail rank 89. Across all families, the worst oracle tail rank is
KDA's 160/512 (31.25%); none of the 25 oracle winners is at or above 50%.

All 25 final oracle guards pass. In particular, all five attention oracles are
selected, compile with 168 registers and zero spill-store, spill-load, or local
bytes, pass the post-compile filter, and are benchmarked. All 25 included GPU
attempts have uncontended monitors; one interrupted GEMM attempt marked
`contended` is retained for provenance but excluded from aggregation.

| Family | Mean case speedup | Total-time speedup | Oracles preserved |
|---|---:|---:|---:|
| GEMM | 4.225x | 4.216x | 5/5 |
| FlashAttention | 1.425x | 1.426x | 5/5 |
| KDA | 3.402x | 3.447x | 5/5 |
| FP8 GEMM | 3.756x | 3.756x | 5/5 |
| Grouped GEMM | 4.556x | 4.553x | 5/5 |
| **Overall** | **3.473x** | **3.320x** | **25/25** |

The completed comparison is in
`experiments/results/b200-memory-combined-strict-alpha50-gpu0-1-g4-spill-detect-20260919/comparison.json`.

Run another strict fraction with, for example:

```bash
python -m experiments.combined --gpus 0 1 --alpha 0.5 --output /tmp/tiletune-combined
```

Reproduce the CPU-only replay with:

```bash
python -m experiments.replay_memory \
  --study experiments/results/b200-v10-five-family-top20-20260917 \
  --output /tmp/tiletune-b200-memory-replay \
  --runs flash_attention gemm gemm_fp8 grouped_gemm kda-periodic-v2
```

The output contains a row for every workload, the oracle's display position and
conservative tail rank, SHA-256 provenance for both archived analysis and oracle
files, and aggregate 20%/50% coverage.
