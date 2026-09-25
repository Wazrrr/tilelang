# KDA experiments

The named suite calls the token-parallel intra-chunk example
[`chunk_intra_token_parallel.py`](../../examples/kda/chunk_intra_token_parallel.py)
(`tilelang_chunk_kda_fwd_intra_token_parallel`) directly. Q/K use BSHD and the
gates/beta use (B,S,H) and (B,S,H,D) layouts; the example derives the intra-chunk
coefficients with one query row per token and parallel hidden tiles.

`kernel.py` supplies input generation, the numerical reference and output
indices. The TileLang program is built directly by the example. `spaces.py`
defines the single configuration pool using the example's parameter names;
`cases.py` defines shapes. Planning imports only the Python standard library.

TileTune reads the example's actual loop and ownership: the query/key tile is
re-read for every intra-chunk iteration, while the gate and beta vectors are
shared consumers. Profile version 8 measures both MMA and WGMMA on Hopper and
applies their rates separately. Carver scores this kernel with a dedicated
`kda_intra_token_parallel` traffic/wave adapter: because the intra kernel issues
no tensor-core MMA, the model uses its `(token, head-block)` CTA domain and the
gated `Aqk = Q K^T` / `Akk = (beta K) K^T` memory ledger instead of a
tensorized GEMM.

## Cases

All named-suite cases use BF16 with `dim=128`, `chunk_size=64` and
`sub_chunk_size=16`. Smoke uses the first development case.

| Split | Case | Parameters |
| --- | --- | --- |
| Training | `kda_intra_train_a` | batch=1, heads=16, sequence=1024 |
| Training | `kda_intra_train_b` | batch=2, heads=16, sequence=2048 |
| Validation | `kda_intra_validation` | batch=1, heads=48, sequence=4096 |
| Development | `kda_intra_short` | batch=1, heads=32, sequence=1024 |
| Development | `kda_intra_medium` | batch=1, heads=64, sequence=2048 |
| Development | `kda_intra_regular` | batch=1, heads=32, sequence=4096 |
| Development | `kda_intra_batched` | batch=2, heads=32, sequence=2048 |
| Development | `kda_intra_long` | batch=1, heads=64, sequence=8192 |
| Final | `kda_intra_short` | batch=1, heads=32, sequence=2048 |
| Final | `kda_intra_medium` | batch=1, heads=64, sequence=4096 |
| Final | `kda_intra_regular` | batch=1, heads=32, sequence=8192 |
| Final | `kda_intra_batched` | batch=2, heads=32, sequence=4096 |
| Final | `kda_intra_long` | batch=1, heads=64, sequence=16384 |

Every sequence is a whole number of 64-token chunks, and every chunk is a whole
number of 16-token sub-chunks; no artificial tails are included.

## Configuration space

Every case uses the same **234-config `expanded` pool**:

| Parameter | Values |
| --- | --- |
| `block_H` | 1, 2, 3, 4, 7, 8, 10, 12, 13, 14, 15, 16 |
| `num_stages` | 0 |
| `threads` | 32, 64, 128, 256, 512, 1024 |
| `block_DK` | 4, 8, 16, 32, 64, 128 |

Only `block_H`/`threads` combinations that lower to a benchmark-safe layout are
kept; combinations that hang at benchmark time and wedge the CUDA context are
removed (`block_H` 5, 6, 9 and 11 have no safe layout and are absent entirely).
`num_stages` is fixed at 0 because deeper pipelines can hang, and the
slimmer `block_DK` slices (4, 8, 16) are restricted to power-of-two `block_H`
tiles because other layouts hang. The 24 `num_stages=0` configs from the
example's autotune grid are included. There is one `expanded`
space with no alternative `current`, `large` or `exhaustive` presets, no cap,
protected subset, target-dependent domain or structural prefilter. Every
declared candidate is attempted in a full sweep; compilation and correctness
failures remain recorded. Counts describe candidate configs, not a guarantee of
that many valid or distinct compiled programs. Explicit CUDA/HIP configs must
select members of this pool. Smoke and development budgets select original
indices without changing the pool.

## Commands

```bash
# Inspect final cases and complete pools without a GPU.
python -m experiments.kda.tiletune.run --suite final --device ampere --plan

# Analyze, compile and check a 16-config subset.
python -m experiments.kda.tiletune.run --suite smoke --device ampere \
  --output experiments/results/kda/smoke-v1

# Audit the complete development pools on idle devices.
python -m experiments.kda.census --device ampere --config-space expanded \
  --wait-idle --output experiments/results/kda/census-v1
```

The [shared runner](../common/README.md) describes brute-force collection,
contention monitoring, timing provenance and comparison runs. Named final
comparisons use K=20, seeds 123/456/789 and seven shuffled winner checks; they
require a passing development report. Native support needs device validation;
planning a target does not establish that its compiler supports every candidate.
Use a new output directory after source or configuration changes.

## Recorded results

No new GPU sweep has been recorded for this configuration update. Earlier
heuristic JSONs are preserved under
`experiments/results/pre-three-single-pools-20260916/kda/heuristics/`. They
describe old kernels and pools. Current measurements belong in `heuristics/`
with their own config IDs, timings and source provenance.

## System ablations and reusable baselines

```bash
python -m experiments.kda.system.run --variant all --plan
python -m experiments.kda.system.run --variant all \
  --output experiments/results/kda/system-v1
python -m experiments.kda.tiletune.run --suite full --device ampere \
  --baseline-root experiments/results/baselines \
  --output experiments/results/kda/tiletune-revision-a
```

System runs support baseline, pipeline, grouped, multi_gpu and combined modes.
New TileTune output directories reuse verified baseline bundles while the
kernels, pools and measurement environment remain compatible. Baseline XGBoost
uses a fixed seed independently of TileTune repeats. Carver scores KDA with the
intra-chunk traffic/wave adapter (no tensor-core MMA is assumed). See the
[workflow guide](../README.md) for GPU monitoring, baseline identity, artifact
paths and arbitrary-K comparisons.
