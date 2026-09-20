# CUDA branch compatibility review

This is a historical version-2 / space-7 checkpoint. `dev-h200-new` now uses
KDA intra-chunk with a 512-config pool (contract 3 / space 8); see
[the current contract](BENCHMARK_CONTRACT.md). The compatibility and measurement
claims below apply to the reviewed revisions, not to the current KDA operation.

Reviewed on 2026-09-17: `dev-a100` at `419b1b0a`, `dev-h200` at
`158d8052`, and `dev-b200` at `00707d8a`, including local changes. Origin was
fetched; these branch tips already matched their upstreams. The A100 worktree
was created beside the existing H200 and B200 worktrees. Pre-existing H200
edits and benchmark artifacts were preserved. Changes are uncommitted.

## Shared experiment contract

All three branches now share five operations and **25 final workloads**.
KDA retains only chunk output. Each operation has five final shapes, two training
shapes and one validation shape, with disjoint training/validation and final sets.
The [benchmark contract](BENCHMARK_CONTRACT.md) specifies all dtypes, scales,
gates, shapes, model assumptions and scheduling axes.

| Operation | Configurations per shape |
| --- | ---: |
| BF16 GEMM | 3,456 |
| FP8 GEMM | 576 |
| Grouped GEMM | 576 |
| FlashAttention | 576 |
| KDA chunk output | 1,296 |

Every pool exceeds the requested 500-candidate minimum. Pool order, shape splits,
active Carver adapters, core policy code, active templates and shared examples
match. Space version 7 preserves all version 6 candidates for these operations.
The full final matrix contains 32,400 candidate evaluations per device per seed,
before training/validation collection and winner validation.

FP8 uses the BF16 GEMM shape set and explicit FP32 scales per row per 128 K
elements for both operands. Contract version 2 separates this scale layout from
older measurements. Existing artifacts remain historical. Source provenance
includes the backend choice and active examples. The four unused KDA component
adapters and their experiment registrations were removed; their original
standalone examples were restored.

## Allowed backend differences

- A100 stores E4M3 inputs and converts them exactly to BF16 for tensor-core
  compute. H200/B200 use native E4M3 compute. Scale buffers and outputs match.
- B200 retains its TCGen05 BF16 GEMM and specialized attention builders.
- Architecture capacities affect Carver estimates. Shared equations do not imply
  identical predictions or generated instructions.
- The unused B200 grouped-MXFP8 template remains available; the active grouped
  BF16 template matches. Branch-specific study orchestration is retained.

The common FP8 builder may be slower than the former specialized builders.
Carver's FP8 adapter uses explicit traffic and storage estimates; instruction
timing remains an approximation. No ranking-accuracy improvement is claimed.
KDA uses the same chunk-output builder, pipeline-safe casts, reference and
1,296-candidate pool across all three branches.

## Validation

The standard-library-only comparison reports `compatible: true`, with no
unexpected differences:

```bash
python -m experiments.compare_branches . ../tilelang-dev-a100 ../tilelang-dev-b200
```

Its only allowed differences are A100's FP8 compute dtype and B200's GEMM and
attention builders. The active suite contains only `kda_chunk_o` for KDA.
After reducing KDA, 138 focused contract, pool and planning checks passed in
each worktree. Retained shape definitions, ordered pools and Carver adapter
hashes match the versions before the reduction. H200 integration passed 85
checks covering chunk-output correctness (including tail dimensions), baseline
reuse and metadata/model analysis. Representative configurations for all 25
active final workloads receive TileTune scores, with no expected failures.
Ruff and whitespace checks passed for the updated code.

Earlier validation covered H200 numerical checks for all 25 active final
workloads, native/emulated FP8 with distinct per-row scales, and one newly added
candidate per active operation. The expanded FP8 tile cross-compiled for SM80
and SM100a. An additional FP8 candidate (M tile 192, N tile 64, stage 1, 128
threads) failed numerical checking at M=N=256, K=512; its stage-0 counterpart
passed. Failed candidates remain recorded failures in sweeps.

Only H200 hardware was available. Tests in the A100/B200 checkouts do not
establish native-device correctness or performance. No exhaustive configuration
sweep or performance validation was performed. Declared pool sizes do not
promise that every candidate compiles, produces distinct code or passes checks.
