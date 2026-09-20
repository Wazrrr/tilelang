# BF16 KDA intra-chunk

The active KDA operation is `kda_chunk_intra_token_parallel`, built directly
through [the example](../../examples/kda/chunk_intra_token_parallel.py)'s
`tilelang_chunk_kda_fwd_intra_token_parallel.jit_impl.get_tir`.
The implementation, cases, pool and reference follow `dev-b200-tiletune` at
`133bfa89506b3e763e562cd51fcc471faf98dc90`. H200 retains its own compiler backend
and unified TileTune model.

Each CTA owns one token and a tile of heads. It computes query/key coefficients
within that token's causal sub-chunk, and strictly causal beta-weighted key/key
coefficients. Aqk is zero outside the corresponding sub-chunk block; Akk has a
zero diagonal. This is the token-parallel intra stage, not the complete KDA
forward pass.

## Inputs and outputs

All named cases fix head dimension 128, chunk size 64, sub-chunk size 16 and
query scale `128**-0.5`.

| Tensor | Dtype | Shape |
| --- | --- | --- |
| Q, K | BF16 | (B,S,H,128) |
| Cumulative gates | FP32 | (B,S,H,128) |
| Beta | BF16 | (B,S,H) |
| Aqk output | BF16 | (B,S,H,64) |
| Akk output | BF16 | (B,S,H,16) |

Accumulation is FP32. Seeded input preparation uses cumulative log-sigmoid gates
reset at chunk boundaries; the kernel applies `exp2(g_i - g_j)`. Gate
preparation is outside the timed kernel. The independent Torch reference uses
sub-chunk matrix products with a stable exponential factorization.

## Final workloads

| Workload | B | S | H | Q/K/gate base shape |
| --- | ---: | ---: | ---: | --- |
| kda_intra_short | 1 | 2048 | 32 | (1,2048,32,128) |
| kda_intra_medium | 1 | 4096 | 64 | (1,4096,64,128) |
| kda_intra_regular | 1 | 8192 | 32 | (1,8192,32,128) |
| kda_intra_batched | 2 | 4096 | 32 | (2,4096,32,128) |
| kda_intra_long | 1 | 16384 | 64 | (1,16384,64,128) |

`cases.py` also defines five development cases, two training cases and one
validation case; training/validation shapes are disjoint from the final shapes.

## Configuration pool

Use the proposed **512-config** Cartesian product without modification:

| Parameter | Values |
| --- | --- |
| block_H | 1–16 inclusive |
| num_stages | 0–7 inclusive |
| threads | 32, 64, 128, 256 |

The example's original grid is `block_H={1,2,4,8}`,
`num_stages={0,1,2,3}`, `threads={128,256}`: **32 configs**.
Every original config and the example's default are included. The expanded
pool is 16 times the original grid. Config IDs and ordering are deterministic;
compile/correctness failures remain outcomes, without pruning or replacement.
Strict alpha=0.5 has a maximum budget of **256** per workload.

## Execution and provenance

`kernel.py` supplies inputs and calls the example; `reference.py` defines the
independent numerical check. Both output tensors are checked. Source fingerprints
name the intra example. Contract version 3 / space version 8 replace the
chunk-output operation; its saved results and heuristics remain historical and
cannot serve as intra-chunk oracles.

Carver's existing `KDAChunkTemplate` describes chunk output. Intra-chunk Carver
requests report unsupported instead of using that unrelated model. This does
not affect the three brute-force/TileTune experiments.

```bash
python -m experiments.kda.tiletune.run --suite full --device hopper --plan
python -m experiments.kda.system.run --plan
```

See the [three-run plan](../H200_THREE_RUN_PLAN.md) for the 128-worker H200 study.

## H200 validation

The expanded pool was checked against both the pre-migration example grid and
the B200 pool: all 32 original configs are present, and the B200 cases/pool match.
Both outputs passed H200 correctness checks for all five final shapes, plus
FP16/BF16 head-tail and mask checks. Eight original-pool configs compiled in one
group and passed both numerical checks and the H200 post-compile resource policy.

CPU analysis of all 2,560 final KDA candidates produced finite memory scores;
strict alpha=0.5 selected 256 per workload. The unified analyzer was unchanged.
These checks establish integration and score coverage. Exhaustive oracle
collection and oracle-retention validation remain part of the planned study.
