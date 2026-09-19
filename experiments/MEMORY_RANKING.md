# Simplified TileTune memory ranking

The `simplify-tiletune-memory` worktree adds `ranking_metric="memory"`. It scores
all **29,200 configurations** in the frozen H200 25-case study. All **25 oracle
winners are eligible and within the first 50% of their pools**; 19 are within
the first 20%. No kernel, configuration pool, oracle timing, or GPU benchmark was
changed or rerun.

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

order by (byte_work * waves, memory_events * waves, original_index)
```

The one-CTA-per-SM wave count is a fixed ordering convention. It accounts for
grid size without predicting physical occupancy. The second key prefers fewer
memory operations when byte work is exactly tied. It does not estimate their
duration. There are no fitted weights, kernel-family branches, oracle labels,
compute throughput profiles, or inferred pipeline-overlap equations in the score.

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
| GEMM | 2,304 | 11,520/11,520 | 5/5 | 5/5 | 271/2,304 = 11.76% |
| FlashAttention | 320 | 1,600/1,600 | 1/5 | 5/5 | 132/320 = 41.25% |
| KDA | 720 | 3,600/3,600 | 5/5 | 5/5 | 2/720 = 0.28% |
| FP8 GEMM | 2,304 | 11,520/11,520 | 5/5 | 5/5 | 363/2,304 = 15.76% |
| Grouped GEMM | 192 | 960/960 | 3/5 | 5/5 | 96/192 = 50.00% |

Uniform 50% budgets, rounded up against each complete pool, reach all 25 oracles.
The limiting case is `grouped_gemm_decode`. The other cases beyond 20% are the
four longer/noncausal attention cases and `grouped_gemm_down_aligned`.

Small-budget performance is a material tradeoff:

| Budget | Original exact hits | Memory exact hits | Original geomean Oracle@K | Memory geomean Oracle@K |
|---|---:|---:|---:|---:|
| K=20 | 12/25 | 13/25 | 91.45% | 68.21% |
| K=100 | 12/25 | 15/25 | 91.72% | 81.78% |
| 20% of each pool | 12/25 | 19/25 | 91.72% | 79.24% |
| 50% of each pool | 12/25 | 25/25 | 91.72% | 100.00% |

All 25 cases have at least one successful candidate at each listed budget.
Failures consume slots without replacement. The memory heuristic can prioritize
large tiles with poor execution performance because it deliberately omits compute,
spill-service, and overlap timing. It improves oracle reachability but is not a
uniform improvement at small budgets. It remains an explicit opt-in mode;
`pipeline_time` keeps its previous default and equations.

## Implementation and use

The reading path is:

1. [Memory access extraction](../tilelang/tiletune/memory.py): operation regions,
   loop bounds, launch size, and dependencies.
2. [Numerical score](../tiletune_core/memory.py): logical byte/event wave counts.
3. [Ranking](../tiletune_core/ranking.py): memory-event tie break, then original
   index; measured latencies never enter the ordering.
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
profile loading in that mode. Analysis version 25 distinguishes the new reports.
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

Focused tests cover masked-tail accounting, memory-event tie order, frozen
provenance, independence from measured labels, missing memory effects, explicit
resource limits, legacy ranking, and the compiler-independent core. GPU tests
are skipped for this change: **135 tests passed, 5 GPU tests skipped**.
