# H200 CUDA benchmark contract, version 3

`dev-h200-new` retains the four GEMM/attention families from contract version 2
and replaces KDA chunk output with token-parallel intra-chunk from
`dev-b200-tiletune` at `133bfa89`. There are five families, five independently
tuned operations, and five final shapes per operation: **25 final workloads**.
Historical cross-branch alignment is recorded in `BRANCH_COMPATIBILITY.md`;
compare live sources before asserting that another branch shares this contract.

| Family / operation | Contract | Configurations per case |
| --- | --- | ---: |
| BF16 GEMM | BF16 A/B/C; FP32 accumulation; A=(M,K), B=(N,K) | 3456 |
| FP8 GEMM | E4M3 A/B; FP32 scales per row and 128 K elements on **both** operands; FP32 partial/total accumulation; BF16 C | 576 |
| Grouped GEMM | Packed BF16 A/B/C; FP32 accumulation; fixed 64-row scheduling tiles | 576 |
| FlashAttention | BF16 forward; FP32 online softmax and accumulators | 576 |
| KDA intra-chunk | BF16 Q/K/beta/Aqk/Akk; FP32 gates and accumulators | 512 |

Counts are declared candidates, not guaranteed compiler successes. Every method
uses the same ordered pool. Invalid configurations retain their failed outcomes;
there is no architecture-specific pruning or replacement of failed candidates.
Attention M tiles are 32/64/128 on all branches: larger tiles were excluded from
the common domain because the native SM100 implementation can hang on them.

## Scheduling domains, space version 8

Every pool contains at least 500 candidates and includes its active example's original pool.
Each row below is a Cartesian product of scheduling choices. Ranges are inclusive; M/N/K and block_H
refer to scheduling tiles, not workload dimensions.

| Operation | Tile choices | Threads | Stages | Other fixed/tuned settings |
| --- | --- | --- | --- | --- |
| BF16 GEMM | M,N each: 32,64,96,128,192,256; K: 16,32,48,64 | 128,256 | 0–5 | Rasterization on/off |
| FP8 GEMM | M,N each: 32,64,96,128,192,256 | 128,256 | 0–7 | K=128; fixed row scales |
| Grouped GEMM | N: 32,64,96,128,192,256; K: 16,32,48,64,96,128 | 128,256 | 0–7 | M=64; fixed metadata |
| Attention | M: 32,64,128; N: 16,32,48,64,80,96,112,128,160,192,224,256 | 128,256 | 0–7 | Workload causal flag |
| KDA intra-chunk | block_H: 1–16 | 32,64,128,256 | 0–7 | Fixed head dimension=128, chunk=64, sub-chunk=16 |

Version 8 replaces the KDA operation and pool; the other four pools are unchanged.
The 512-config KDA pool contains all 32 original intra example configs: block_H
in {1,2,4,8}, stages in {0,1,2,3}, threads in {128,256}. Chunk-output bundles
cannot serve as intra-chunk baselines. Kernel contract version 3 records this
semantic change. Smoke/development still select their declared budgeted subsets;
final/full use all candidates. The 25 final workloads contain 28,480 candidate
evaluations per device per repeat, before winner validation and training.

Baseline collections containing KDA use contract version 3. Collections for
the four unchanged families retain version 2, preserving compatible baselines.

## Shapes

Dense BF16 and FP8 share every development/training/validation/final shape.
The aligned domain uses M divisible by 32 and N/K divisible by 128; masked
loads/stores handle a tile larger than a small M. Final (M,N,K) shapes are:

- (256,4096,4096): small M;
- (1024,4096,4096): rectangular projection;
- (1024,4096,14336): long reduction;
- (4096,4096,4096): square;
- (4096,14336,4096): rectangular expansion.

Grouped final workloads use 4 skewed decode groups [1,2,4,8], 8 balanced groups
[32]×8, 4 balanced groups [128]×4, 3 balanced down-projection groups [256]×3,
and skewed groups [63,77,111,280]. Both B layouts remain explicit in cases.py.

Attention final (B,H,S,D,causal) shapes are (1,32,512,64,true),
(2,16,2048,64,true), (1,32,4096,128,false), (1,32,4096,128,true), and
(1,16,8192,128,true).

KDA intra-chunk uses the final (B,H,S) shapes: (1,32,2048),
(1,64,4096), (1,32,8192), (2,32,4096), and (1,64,16384). Head dimension=128,
chunk size=64 and sub-chunk size=16 are fixed. Each operation has two training cases and one validation case, disjoint from its five final cases.
The machine-readable source is each family's cases.py and the frozen manifest.

## Allowed backend implementations

- A100 converts E4M3 storage values exactly to BF16 for tensor-core GEMM.
  This is FP8 storage emulation, not native FP8 throughput. Its public scale
  buffers and pool are identical to H200/B200.
- H200/B200 use native E4M3 operands in the common block-scaled example. The old
  Hopper 128×128 B-scale layout and SM100 packed-UE8M0 persistent pool are no
  longer the experiment contract; their standalone examples remain available.
- B200 retains its SM100 TCGen05 dense GEMM and attention examples. A100/H200
  use the advanced GEMM and portable BSHD attention examples.
- Grouped GEMM retains the common concatenated example. H200 KDA now uses
  the token-parallel intra example imported from `dev-b200-tiletune`.

`experiments/backend.py` makes the worktree's FP8 compute choice explicit.
Backends do not introduce tuning knobs, change mathematical shapes, or switch
scale granularity. FP8 always uses FP32 scales of shapes (M,K/128) and (N,K/128).

## KDA intra-chunk semantics

Only `kda_chunk_intra_token_parallel` is active. Q/K are BF16 (B,S,H,128),
gates are FP32 with the same shape, and beta is BF16 (B,S,H). Outputs are BF16
Aqk (B,S,H,64) and Akk (B,S,H,16); accumulation is FP32. Head dimension=128,
chunk=64, sub-chunk=16 and query scale=128^-0.5 are fixed.

The example computes causal query/key coefficients within each 16-token
sub-chunk and strictly causal beta-weighted key/key coefficients. Aqk entries
outside that sub-chunk block and Akk's diagonal are zero. Gates are cumulative
log-sigmoid values reset at chunk boundaries; the kernel evaluates exp2 of
gate differences. Gate preprocessing is outside timing. The independent Torch
reference uses a stable exponential factorization and sub-chunk matrix products.

Results cover this token-parallel intra stage. Inter-solve, WY, recurrent state
updates and chunk output remain standalone examples outside the active suite.
See [KDA workloads and pool](kda/README.md) for the complete tensor contract.

## Models and provenance

Carver's existing dense and
grouped GEMM retain the existing policy. FP8's shared traffic/wave adapter counts
explicit scale loads, operand storage/conversion and two FP32 accumulator tiles.
The existing KDAChunkTemplate models chunk output; intra-chunk Carver requests
report unsupported. TileTune analyzes the intra PrimFunc with its unified model.
Unsupported scheduling remains an explicit model diagnostic; compatibility is
not evidence of ranking quality.

Contract version 3 separates new runs from old measurement bundles. Source
fingerprints include backend selection and the active kernels and examples.
Old artifacts remain historical and must not be relabeled as version 3 results.

Run the standard-library-only comparison from any worktree:

```bash
python -m experiments.compare_branches . ../tilelang-dev-a100 ../tilelang-dev-b200
```

It checks shapes, pool hashes, active Carver adapter hashes and shared examples,
allowing only the backend differences above. GPU performance still requires an
observed uncontended run on the intended device. Cross-compilation establishes
compilation only.
