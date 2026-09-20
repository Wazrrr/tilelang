> The current cross-branch contract is [BENCHMARK_CONTRACT.md](BENCHMARK_CONTRACT.md): five families, five operations and 25 final workloads. It supersedes older pool and FP8/KDA descriptions below.

# Current workload and model contracts

The active matrix has five families and twenty-five final cases: FP16 GEMM, FP8 GEMM,
FlashAttention, chunk-KDA output and grouped GEMM. Softmax source is archived in
`results/archive/softmax-source-20260917/`; existing measurements keep their
original meaning. All experiment kernels call their authoritative example
builders. The FP8 family uses FP32 accumulation and FP8 output. Its explicit
`tl.disable_wgmma=True` compiler policy keeps the existing per-element accuracy
check; long WGMMA accumulation failed that check on the final H200 shapes.
Every method uses this policy, and TileTune uses the measured MMA profile.

## TileTune

Analysis version 24 separates the following inputs and assumptions:

- Matrix work comes from the kernel's lowered matrix operation, including padded
  tiles and both KDA products. Attention retains its causal iteration bounds,
  online normalization and reductions.
- Grouped GEMM declares its actual integer sizes, offsets and padded offsets via
  `TileTuneConfig.input_values`. Keys are PrimFunc parameter indices; values are
  read-only integer vectors with exact dtype/shape checks. The experiment runner
  verifies actual inputs before execution. Other API callers must do the same.
- Metadata is resolved only in the analysis view. Small scalar setup loops are
  evaluated under this contract; matrix loops remain symbolic. Original metadata
  loads still contribute traffic and scalar work. The compiled PrimFunc is intact.
- Ragged groups have separate CTA work classes. Full padded GEMM work, A loads
  that cross a group boundary, the packed allocation's final mask, and every
  group's output mask are counted separately. Interacting row and column tails
  are partitioned together in CUDA's x-fastest launch order; repeated batch/head
  patterns remain compressed. The bounded analysis leaves excessively complex
  or unresolved domains unscored rather than assuming uniform CTA work.
- Profile version 6 measures both WGMMA and MMA on Hopper. Matrix rates are
  selected per operation by instruction, A dtype, B dtype and accumulator dtype.
  Older cached profiles remain readable but cannot supply missing signatures.
  The 48-row KDA tail kernel needs MMA measurements even on Hopper.

These changes make all twenty-five workloads analyzable with matching profiles and
supported configurations. They do not guarantee every configuration has a score.
Unresolved compiler pipeline plans, resource limits, missing measurements and
unsupported instructions remain explicit diagnostics. In particular, positive-
stage grouped/KDA schedules are not inferred from a stage count alone.

Byte/FLOP tests check executed work independently of the timing equations.
Latency still uses measured effective primitive rates, estimated occupancy and
supported scheduling models. Exact work counts do not make predicted latency an
exact hardware simulation. Synthetic test profiles are never performance results.

## Carver

| Family | Template | Retained limitations |
| --- | --- | --- |
| FP16/BF16 GEMM | Existing `MatmulTemplate` | Original traffic/wave priority; no rasterization timing |
| FP8 GEMM | `FP8MatmulTemplate` | FP32 accumulation; E4M3FN lowers to Carver's tensorizable E4M3 spelling |
| Attention | `FlashAttentionTemplate` | Scaling, masking, stable softmax, probability cast and both GEMMs are retained |
| Grouped GEMM | `GroupedMatmulTemplate` | The padded CTA domain is modeled; metadata lookup and interleaved CTA scheduling are omitted |
| KDA intra-chunk | Unsupported | `KDAChunkTemplate` describes the retired chunk-output operation |

Adapters evaluate the complete experiment pool through the existing policy.
Policy scores and feasibility rules are unchanged. The final attention pools,
ragged grouped case and historical 48-row KDA chunk-output case can have no feasible candidates. These
save every rejection and report `model_unavailable`; they receive no replacement
ranking. SM100 uses Carver's SM90 tensorization vocabulary while retaining
visible-device capacities.

Old baseline bundles remain immutable. To evaluate the newly added templates,
explicitly collect a new baseline bundle; unchanged oracle tables may be reused
with their original measurement provenance. Use `compare_results.sh --top-k 20`
only with complete, matching oracle and ranking artifacts.
