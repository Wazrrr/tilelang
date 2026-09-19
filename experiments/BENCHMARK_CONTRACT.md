# Common CUDA benchmark contract, version 2

`dev-a100`, `dev-h200`, and `dev-b200` share the same families, development,
training, validation and final shapes, ordered configuration pools, and Carver
adapter equations. There are five families, five independently tuned operations,
and five final shapes per operation: **25 final workloads**.

| Family / operation | Contract | Configurations per case |
| --- | --- | ---: |
| BF16 GEMM | BF16 A/B/C; FP32 accumulation; A=(M,K), B=(N,K) | 3456 |
| FP8 GEMM | E4M3 A/B; FP32 scales per row and 128 K elements on **both** operands; FP32 partial/total accumulation; BF16 C | 576 |
| Grouped GEMM | Packed BF16 A/B/C; FP32 accumulation; fixed 64-row scheduling tiles | 576 |
| FlashAttention | BF16 forward; FP32 online softmax and accumulators | 576 |
| KDA output | BF16 Q/V/A/state/output; FP32 accumulators | 1296 |

Counts are declared candidates, not guaranteed compiler successes. Every method
uses the same ordered pool. Invalid configurations retain their failed outcomes;
there is no architecture-specific pruning or replacement of failed candidates.
Attention M tiles are 32/64/128 on all branches: larger tiles were excluded from
the common domain because the native SM100 implementation can hang on them.

## Scheduling domains, space version 7

Every pool contains at least 500 candidates and includes all version 6 candidates.
Each row below is a Cartesian product of scheduling choices. Ranges are inclusive; M/N/K and DK/DV
refer to scheduling tiles, not workload dimensions.

| Operation | Tile choices | Threads | Stages | Other fixed/tuned settings |
| --- | --- | --- | --- | --- |
| BF16 GEMM | M,N each: 32,64,96,128,192,256; K: 16,32,48,64 | 128,256 | 0–5 | Rasterization on/off |
| FP8 GEMM | M,N each: 32,64,96,128,192,256 | 128,256 | 0–7 | K=128; fixed row scales |
| Grouped GEMM | N: 32,64,96,128,192,256; K: 16,32,48,64,96,128 | 128,256 | 0–7 | M=64; fixed metadata |
| Attention | M: 32,64,128; N: 16,32,48,64,80,96,112,128,160,192,224,256 | 128,256 | 0–7 | Workload causal flag |
| KDA output | DK: 16,32,48,64,96,128; DV: 16,32,48,64,80,96,112,128,160,192,224,256 | 64,128,256 | 0–5 | Fixed chunk=64 |

Version 7 changes pool membership and indices; configuration hashes for retained
candidates remain stable. Stored bundles with the old full pools cannot be reused
as complete version 7 baselines. Kernel contract version 2 remains unchanged.
Smoke/development still select their declared budgeted subsets; final/full use
all candidates. The 25 final workloads contain 32,400 candidate evaluations per
device per repeat, before winner validation and training/validation collection.

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

KDA chunk output uses the final (B,H,S) shapes: (1,32,2048),
(1,64,4096), (1,32,8192), (2,32,4096), and (1,64,16384). DK=DV=128
and chunk size=64 are fixed. Each operation has two training cases and one validation case, disjoint from its five final cases.
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
- Grouped GEMM and KDA chunk output use the same examples on all three
  branches. The KDA output expression retains Ampere's pipeline-safe casts.

`experiments/backend.py` makes the worktree's FP8 compute choice explicit.
Backends do not introduce tuning knobs, change mathematical shapes, or switch
scale granularity. FP8 always uses FP32 scales of shapes (M,K/128) and (N,K/128).

## KDA chunk-output semantics

Only `kda_chunk_o` is part of the experiment family. Its Q, V, attention and
per-chunk state inputs and output are BF16; accumulators are FP32. DK=DV=128
and chunk size=64 are fixed for every named workload. Gates are FP32 base-2
cumulative log gates, reset at chunk boundaries, and the query scale is DK^-0.5.
Gate preprocessing is outside the timed kernel. State is supplied as
(B,S/64,H,128,128); the chunk-output kernel consumes it without updating it.
The independent Torch reference preserves the example's BF16 rounding points.

The pool tunes key/value tiles, thread count and pipeline stages. Results measure
chunk output only. Intra, inter-solve, WY and recurrent state updates are outside
the active suite; the benchmark does not measure a complete FlashKDA pipeline.
The repository's standalone examples remain available.

## Models and provenance

Carver uses the same adapter equations and ordering on all branches. Dense and
grouped GEMM retain the existing policy. FP8's shared traffic/wave adapter counts
explicit scale loads, operand storage/conversion and two FP32 accumulator tiles.
KDA chunk output retains its shared Carver template and reference. TileTune can
score the representative configurations tested for all five active operations.
Unsupported scheduling remains an explicit model diagnostic; compatibility is
not evidence of ranking quality.

Contract version 2 separates new runs from old measurement bundles. Source
fingerprints include backend selection and the active kernels and examples.
Old artifacts remain historical and must not be relabeled as version 2 results.

Run the standard-library-only comparison from any worktree:

```bash
python -m experiments.compare_branches . ../tilelang-dev-a100 ../tilelang-dev-b200
```

It checks shapes, pool hashes, active Carver adapter hashes and shared examples,
allowing only the backend differences above. GPU performance still requires an
observed uncontended run on the intended device. Cross-compilation establishes
compilation only.
