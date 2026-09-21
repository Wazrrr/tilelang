# H200 CUDA benchmark contract, version 4

`dev-h200-new` defines five kernel families with five final workloads each.
KDA uses token-parallel intra-chunk; FP8 calls the original unscaled E4M3
example with FP32 accumulation and E4M3 output. Other branches are outside
this contract.

| Family / operation | Contract | Configurations per case |
| --- | --- | ---: |
| BF16 GEMM | BF16 A/B/C; FP32 accumulation; A=(M,K), B=(N,K) | 576 |
| FP8 GEMM | E4M3 A/B/C; FP32 accumulation; A=(M,K), B=(N,K) | 576 |
| Grouped GEMM | Packed BF16 A/B/C; FP32 accumulation; fixed 64-row scheduling tiles | 576 |
| FlashAttention | BF16 forward; FP32 online softmax and accumulators | 512 |
| KDA intra-chunk | BF16 Q/K/beta/Aqk/Akk; FP32 gates and accumulators | 645 |

Active pools retain only configurations compiled successfully for every final
workload in their family, targeting SM90a. Certificates in `compilation/` record
NVCC and native-library versions, source hashes, original-grid inclusion, and
per-workload evidence. This proves device compilation, not GPU execution,
numerical correctness, post-compile-filter acceptance, or oracle retention.

## Scheduling domains, space version 9

Every active pool contains more than 500 configurations and includes its example's original pool.
Rows below describe candidate axes before measured compilation failures are removed. M/N/K and block_H
refer to scheduling tiles, not workload dimensions.

| Operation | Tile choices | Threads | Stages | Other fixed/tuned settings |
| --- | --- | --- | --- | --- |
| BF16 GEMM | M: 64,128,256; N: 32,64,96,128,192,256; K: 32,64 | 128,256 | 0–3 | Rasterization on/off |
| FP8 GEMM | M,N each: 64,128,256; K: 32,64 | 128,256 | 0–7 | Rasterization on/off |
| Grouped GEMM | N: 32,64,96,128,192,256; K: 16,32,48,64,96,128 | 128,256 | 0–7 | M=64; fixed metadata |
| Attention | M: 32,64,128,256; N: 16–256, step 16 | 128,256 | 0–7 | Workload causal flag |
| KDA intra-chunk | block_H: 1–16 | 32,64,128,256 | 0–15 | Fixed head dimension=128, chunk=64, sub-chunk=16 |

The original GEMM and FP8 example grids contain 288 configurations each; the
KDA intra example contains 32. The active pools must contain every original
configuration. Compilation qualification and numerical correctness are separate.
Smoke/development use recorded subsets; final/full use the complete pools.

FP8 contract version 4 and pool version 9 invalidate incompatible cached
measurements. Source fingerprints identify the actual example and adapter.

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

## H200 implementations

All five experiments use their authoritative example builders. FP8 requires
native FP8 support (SM89 or newer); the H200 compilation target is SM90a.
No emulated FP8 path is used. The FP8 pool restores both rasterization values
and block_K=32/64 from the original example.

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

Results cover the token-parallel intra stage. See [KDA workloads and pool](kda/README.md).

## Models and provenance

TileTune memory scoring remains kernel-independent, with diagnostics opt-in.
Post-compile checks remain separate and require fresh oracle validation for
changed contracts. Carver changes are deferred; unsupported adapters report
their support boundary explicitly. Existing measurement bundles require
matching kernel, workload, target, and pool identities before reuse.
