# Simplified TileTune memory ranking on B200

The `dev-b200-tiletune` worktree adds the family-independent
`ranking_metric="memory"` path from `dev-h200-new`, with one B200 adjustment:
IR pipeline depth orders candidates only when logical byte-waves are identical.
This keeps the model small and avoids fitted compute, occupancy, spill, and
family-specific timing rules.

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
score = logical_byte_waves * 65536 + (65535 - pipeline_depth)
display key = (score, memory_events * waves, original_index)
rank = last position occupied by an equal-score group
```

The radix exceeds the allowed pipeline-depth range, so one fewer byte-wave is
always better than any pipeline-depth difference. Memory events and original
index stabilize output only. They never justify pruning within a primary-score
tie. Runtime top-K selection therefore expands through the complete boundary
group and reports the excess over the requested budget.

The score counts logical, padded accesses. It is not a cache, coalescing,
transaction, bandwidth, compute, or physical-occupancy model. Dependency and
storage reports remain available for inspection and explicit resource policies,
but they do not become an implicit timing formula.

## Frozen-pool replay

The offline replay used the completed
`b200-v10-five-family-top20-20260917` study. It reconstructs score inputs from
the archived TileTune operation facts, freezes every ranking, then opens the
hash-verified exhaustive oracle only to evaluate the winner's rank. It does not
compile kernels, execute kernels, change the pools, or run GPU measurements.

| Family | Cases | Pool per case | Within 20% | Within 50% | Worst tail-rank share |
|---|---:|---:|---:|---:|---:|
| FlashAttention | 5 | 576 | 5/5 | 5/5 | 10.42% |
| GEMM | 5 | 2,304 | 5/5 | 5/5 | 1.04% |
| FP8 GEMM | 5 | 533 | 5/5 | 5/5 | 16.70% |
| Grouped GEMM | 5 | 576 | 3/5 | 5/5 | 40.28% |
| KDA | 5 | 512 | 0/5 | 5/5 | 31.25% |
| **All** | **25** | — | **18/25** | **25/25** | **40.28%** |

A uniform ceil-rounded 41% budget is sufficient for all 25 fixed-pool oracle
winners; the requested 50% guarantee has margin. The limiting case is
`grouped_gemm_prefill`, whose equal-score group ends at rank 232/576. These are
retrospective results for this sealed pool, not a general performance guarantee.

That frozen replay predates direct bounded-`While` handling. Its archived
operation rows did not retain the enclosing `While` nodes, so the replay could
not prove the persistent scheduler's visit bound. The 16.7% FP8 result happened
to identify the correct stage-six group, but it is not the source of truth for
live selection.

## Live 50% combined audit

The final GPU 0/1 combined run analyzes every live PrimFunc. A positive-step
`While` is scored only when its initialization, limit, unconditional update,
and conservative maximum visit count are provable. Manual pipeline depth is
read from an asynchronous global-to-shared copy whose leading shared axis is
indexed modulo that axis's extent. Other `While` loops remain unknown.

| Family | Pool per case | Final selected range | Full-pool cases | Worst oracle tail rank |
|---|---:|---:|---:|---:|
| GEMM | 2,304 | 1,152–1,184 | 0/5 | 24/2,304 (1.04%) |
| FlashAttention | 576 | 288 | 0/5 | 60/576 (10.42%) |
| KDA | 512 | 256 | 0/5 | 160/512 (31.25%) |
| FP8 GEMM | 533 | 311 | 0/5 | 89/533 (16.70%) |
| Grouped GEMM | 576 | 288 | 0/5 | 42/576 (7.29%) |

FP8 has five finite score groups: 89 stage-six configs followed by four groups
of 111 configs. The requested 267-config cutoff lands in the stage-four group,
so the equal-score tail rule expands selection to 311/533 (58.35%), not the
whole pool. All five FP8 oracle winners have tail rank 89 and all 25 final
oracle guards pass. The completed comparison is in
`experiments/results/b200-memory-combined-gpu0-1-g4-spill-detect-20260919/comparison.json`.

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
