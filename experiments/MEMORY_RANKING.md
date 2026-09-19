# Simplified TileTune memory ranking

The `simplify-tiletune-memory` worktree adds `ranking_metric="memory"`. It scores
all **29,200 configurations** in the frozen H200 25-case study. All **25 oracle
winners are eligible**. With equal primary scores assigned their group's tail
rank, **19/25 are within 20% and 20/25 are within 50%**. No kernel, configuration
pool, oracle timing, or GPU benchmark was changed or rerun.

The source is the completed `h200-25shape-20260917T080241Z` FP16/E4M3 study.
These are retrospective results on that fixed pool, not measurements of the
newer BF16 kernels or a claim of generalization.

## Scoring rule

For each captured global read/write, count its requested tile bytes and loop
visits. Scalar accesses retain their enclosing iteration counts. Use the upper
visit count for a branch-dependent loop. Predicates and tail masks may suppress
accesses; the result is logical work, not measured DRAM traffic.

```text
byte_work = sum(access.bytes * access.visits)
memory_events = sum(access.visits for nonempty accesses)
waves = ceil(grid_blocks / SM_count)

score = byte_work * waves
rank = last position occupied by an equal-score group
display order = (score, memory_events * waves, original_index)
```

The one-CTA-per-SM wave count is a fixed ordering convention. It accounts for
grid size without predicting physical occupancy. Memory events order report
entries within a tie, but do not distinguish their predicted ranks or justify
pruning part of that tie. There are no fitted weights, kernel-family branches,
oracle labels, compute throughput profiles, or inferred pipeline-overlap
equations in the score.

The analyzer retains the dependency graph, shared-memory lifetime information,
and live register-tile estimates. They remain inspectable facts and diagnostics;
uncertainty in a compiler schedule or soft register estimate no longer suppresses
a resolved memory score. Explicit resource policies, established violations, and
post-compile checks remain separate. Unresolved actual memory effects or missing
SM/grid information still produce an explicit unknown score.

This also avoids a problem in the old traffic path: independently bounding the
start and end of a clipped partial tile can expand a 96-wide FP8 read into a
4,096-wide region for each iteration. The new path counts the captured 96-wide
access directly, including the padded final iteration. It does not rewrite the
kernel or claim exact mask-adjusted traffic.

## Results on the fixed pool

| Family | Pool per case | Scored | Oracle hits within 20% | Oracle hits within 50% | Worst oracle position |
|---|---:|---:|---:|---:|---:|
| GEMM | 2,304 | 11,520/11,520 | 5/5 | 5/5 | 352/2,304 = 15.28% |
| FlashAttention | 320 | 1,600/1,600 | 1/5 | 1/5 | 168/320 = 52.50% |
| KDA | 720 | 3,600/3,600 | 5/5 | 5/5 | 40/720 = 5.56% |
| FP8 GEMM | 2,304 | 11,520/11,520 | 5/5 | 5/5 | 384/2,304 = 16.67% |
| Grouped GEMM | 192 | 960/960 | 3/5 | 4/5 | 112/192 = 58.33% |

Uniform **58% budgets rounded up** reach all 25 oracles: the limiting case is
`grouped_gemm_decode`, where `ceil(0.58 * 192) = 112`. Its tail rank itself is
58.33% of the pool. The four longer/noncausal attention cases require 168/320.
This supersedes the earlier claim of 25/25 at 50%, which depended on favorable
within-tie ordering. `grouped_gemm_down_aligned` is also outside the first 20%.

For example, candidates occupying positions 81–112 with one score all receive
`rank=112`, while `position` retains each entry's deterministic display position.
Runtime selection keeps the complete boundary group and reports `budget_excess`
when it exceeds requested K. Offline fixed-budget curves count only complete
groups whose tail rank is at most K; an expanded runtime shortlist is evaluated
separately at its actual selected count. Neither reporting nor selection treats
an original index or memory-event tie break as evidence for pruning.

Small-budget performance is a material tradeoff:

| Budget | Original exact hits | Memory exact hits | Original geomean Oracle@K | Memory geomean Oracle@K | Memory cases with a successful candidate |
|---|---:|---:|---:|---:|---:|
| K=20 | 8/25 | 0/25 | 88.50% | N/A | 0/25 |
| K=100 | 12/25 | 14/25 | 91.72% | 68.86% | 25/25 |
| 20% of each pool | 12/25 | 19/25 | 91.72% | 79.20% | 25/25 |
| 50% of each pool | 12/25 | 20/25 | 91.72% | 84.71% | 25/25 |
| 58% of each pool | 12/25 | 25/25 | 91.72% | 100.00% | 25/25 |

Both columns now use complete primary-score groups and conservative tail ranks.
The original timing model has successful candidates in all 25 cases at every
listed budget. At K=20, the memory mode has no successful candidate within the
complete groups that fit this strict budget, so its geometric mean is undefined.
A runtime request for 20 instead expands the boundary tie and reports the larger
actual attempt count. Failures consume slots without replacement.
The memory heuristic can prioritize
large tiles with poor execution performance because it deliberately omits compute,
spill-service, and overlap timing. It improves oracle reachability but is not a
uniform improvement at small budgets. It remains an explicit opt-in mode;
`pipeline_time` keeps its previous default and equations.

## Implementation and use

The reading path is:

1. [Memory access extraction](../tilelang/tiletune/memory.py): operation regions,
   loop bounds, launch size, and dependencies.
2. [Numerical score](../tiletune_core/memory.py): logical byte/event wave counts.
3. [Ranking](../tiletune_core/ranking.py): shared tail ranks for equal primary
   scores, with full boundary groups retained by selection; measured latencies
   never enter the ordering.
4. [Analysis coordinator](../tilelang/tiletune/engine.py): bypasses specialization
   timing, warp-specialization prediction, pipeline timing, and occupancy for
   memory mode, while preserving storage/resource reporting.

```python
config = TileTuneConfig(
    enabled=True,
    ranking_metric="memory",
    top_k=100,
    device_limits=device_limits,  # Includes the target's SM count.
)
```

Existing AutoTuner calls can use
`.set_tiletune_args(True, ranking_metric="memory", top_k=100)`.
The common experiment CLI also accepts `--metric memory` and skips primitive
profile loading in that mode. Analysis version 26 includes conservative tie
ranks and runtime selection that retains whole boundary groups.
Optional `facts_path` output for this mode uses the explicit `memory.v1` schema
and can be replayed with `score_memory(accesses, grid_blocks, sm_count)`; its
`unknown` field must be checked before treating a partial ledger as complete.

Reproduce the complete ranking comparison without TileLang/TVM or a GPU:

```bash
python -m experiments.replay_memory \
  --study /m-coriander/coriander/ziren/code/tilelang/experiments/results/studies/h200-25shape-20260917T080241Z \
  --output experiments/results/memory-ranking/replay
```

The replay verifies original file hashes, preserves pool order and resource
decisions, freezes each complete ranking, then joins oracle measurements for
evaluation. Symbolic loop visits reuse the saved per-operation bounds. The
scorer accepts resolved memory facts only; the archived configuration and oracle
winner are used for reporting after scoring.

- [Full per-case report](results/memory-ranking/replay/report.md)
- [Rankings, curves and provenance](results/memory-ranking/replay/summary.json)
- [Fresh CPU IR checks and timing samples](results/memory-ranking/fresh-ir-validation.json)
- [CPU verification script](results/memory-ranking/validate_frozen_ir.py)

## Validation and analysis cost

Fresh CPU analysis of **74 distinct configurations**, including every oracle
winner and the pool endpoints for every case, exactly matched the replay's
primary and secondary scores. All 25 winners remained eligible, and every
analyzed PrimFunc was unchanged. The frozen example sources were loaded from
the study archive. No GPU was visible to these validation processes.

Three interleaved CPU-analysis samples per representative winner gave:

| Representative | Existing pipeline analysis | Memory analysis | Speedup |
|---|---:|---:|---:|
| Attention causal | 127.50 ms | 98.62 ms | 1.29× |
| GEMM decode | 20.85 ms | 9.99 ms | 2.09× |
| FP8 GEMM decode | 18.36 ms | 8.39 ms | 2.19× |
| Grouped GEMM aligned | 325.14 ms | 292.08 ms | 1.11× |
| KDA batched | 49.58 ms | 34.25 ms | 1.45× |

These are medians for analysis on already elaborated IR, not GPU performance or
total autotuning cost. Collector/dependency/region analysis remains, particularly
for grouped GEMM. Replay timings in the generated report exclude IR capture and
file I/O and are a different scope.

Focused tests cover masked-tail accounting, memory-event display order, frozen
provenance, independence from measured labels, missing memory effects, explicit
resource limits, legacy ranking, and the compiler-independent core. Initial
memory-mode validation had **135 passes and 5 GPU skips**. The subsequent
conservative-rank and boundary-selection change had **94 passes and 13 GPU
skips**, including exploration, saved-selection accounting, and Carver adapter
compatibility. The standalone softmax check additionally analyzed 24 PrimFuncs;
its eight large-shape candidates all receive tail rank 8/8.
