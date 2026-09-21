# Token-parallel KDA intra experiments

The suite calls
[`tilelang_chunk_kda_fwd_intra_token_parallel`](../../examples/kda/chunk_intra_token_parallel.py)
directly. Q, K, and beta use BF16; cumulative gates and accumulation use FP32;
the `Aqk` and `Akk` coefficient outputs use BF16. Every workload uses DK=128,
chunk size 64, and sub-chunk size 16.

The five development/final cases cover 2K–16K sequences, 32/64 heads, and a
batched shape. Training uses `(B,H,S)=(1,16,1024)` and `(2,16,2048)`; validation
uses `(1,48,4096)`.

## Configuration space

Every case uses the same 513-config B200 pool:

| Parameter | Values |
| --- | --- |
| `block_H` | 1 through 16 |
| `num_stages` | 0 through 8 |
| `threads` | 32, 64, 128, 256 |

At 256 threads, odd `block_H` values above one are excluded because the B200
layout planner rejects them. All other combinations compile, and all 32 native
example autotune configurations are included. The pool has no alternative
presets or budget cap. Carver is explicitly unsupported because its chunk-level
template does not model these token-parallel coefficient semantics.

```bash
python -m experiments.kda.tiletune.run --suite final --device blackwell --plan
python -m experiments.kda.tiletune.run --suite smoke --device blackwell \
  --output experiments/results/kda/smoke-v11
```

Pool changes require fresh oracle identities. Historical measurements from
other KDA stages must not be reused as this kernel's oracle.
