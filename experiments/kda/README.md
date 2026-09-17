# KDA experiments

The named suite calls [KDA chunk output](../../examples/kda/chunk_o.py) (`tilelang_chunk_fwd_o`) directly.
Q/V/G/A/O use BSHD; hidden state is (B,chunks,H,DK,DV). Scaled Q and gated Q each materialize in the input dtype, as in the example.

`kernel.py` supplies input generation, the numerical reference and output
indices. The TileLang program is built directly by the example. `spaces.py`
defines the single configuration pool using the example's parameter names;
`cases.py` defines shapes. Planning imports only the Python standard library.

## Cases

All named-suite cases use FP16. Smoke uses the first development case.

| Split | Case | Parameters |
| --- | --- | --- |
| Training | `kda_chunk_train_a` | batch=1, heads=32, sequence=1024, dim=128, value_dim=128, chunk_size=64 |
| Training | `kda_chunk_train_b` | batch=1, heads=32, sequence=2048, dim=128, value_dim=128, chunk_size=64 |
| Validation | `kda_chunk_validation` | batch=1, heads=32, sequence=4096, dim=128, value_dim=128, chunk_size=64 |
| Development | `kda_chunk_short` | batch=1, heads=64, sequence=1024, dim=128, value_dim=128, chunk_size=64 |
| Development | `kda_chunk_medium` | batch=1, heads=64, sequence=2048, dim=128, value_dim=128, chunk_size=64 |
| Development | `kda_chunk_regular` | batch=1, heads=64, sequence=4096, dim=128, value_dim=128, chunk_size=64 |
| Development | `kda_chunk_batched` | batch=2, heads=64, sequence=2048, dim=128, value_dim=128, chunk_size=64 |
| Development | `kda_chunk_long` | batch=1, heads=64, sequence=8192, dim=128, value_dim=128, chunk_size=64 |
| Final | `kda_chunk_short` | batch=1, heads=64, sequence=2048, dim=128, value_dim=128, chunk_size=64 |
| Final | `kda_chunk_medium` | batch=1, heads=64, sequence=4096, dim=128, value_dim=128, chunk_size=64 |
| Final | `kda_chunk_regular` | batch=1, heads=64, sequence=8192, dim=128, value_dim=128, chunk_size=64 |
| Final | `kda_chunk_batched` | batch=2, heads=64, sequence=4096, dim=128, value_dim=128, chunk_size=64 |
| Final | `kda_chunk_long` | batch=1, heads=64, sequence=16384, dim=128, value_dim=128, chunk_size=64 |

The dimensions and chunk size match the repository's serving-oriented KDA
examples. All cases use complete 64-token chunks; no artificial tail dimensions
are included.

## Configuration space

Every case uses the same complete **720-config `expanded` pool**:

| Parameter | Values |
| --- | --- |
| `block_DK` | 16, 32, 48, 64, 96, 128 |
| `block_DV` | 16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256 |
| `num_stages` | 0, 1, 2, 3, 4 |
| `threads` | 128, 256 |

All 90 configs from the example's autotune grid are included: this pool
is exactly 8 times larger. `block_S` stays equal to the workload's chunk size.
Only the chunk-output example is used; recurrent KDA and the independent
row/causal-tile rewrite have been removed from the experiment interface.

Space version 5 has no alternative `current`, `large` or `exhaustive` presets
for this family. There is no cap, protected subset, target-dependent domain or
structural prefilter. Every declared candidate is attempted in a full sweep;
compilation and correctness failures remain recorded. Counts describe candidate
configs, not a guarantee of that many valid or distinct compiled programs.
Explicit CUDA/HIP configs must select members of this pool. Smoke and
development budgets select original indices without changing the pool.

## Commands

```bash
# Inspect final cases and complete pools without a GPU.
python -m experiments.kda.tiletune.run --suite final --device hopper --plan

# Analyze, compile and check a 16-config subset.
python -m experiments.kda.tiletune.run --suite smoke --device hopper \
  --output experiments/results/kda/smoke-v5

# Audit the complete development pools on idle devices.
python -m experiments.kda.census --device hopper --config-space expanded \
  --wait-idle --output experiments/results/kda/census-v5
```

The [shared runner](../common/README.md) describes brute-force collection,
contention monitoring, timing provenance and comparison runs. Named final
comparisons use K=20, seeds 123/456/789 and seven shuffled winner checks; they
require a passing development report. Native support needs device validation;
planning a target does not establish that its compiler supports every candidate.
Use a new output directory after source or configuration changes.

## Recorded results

No new GPU sweep has been recorded for this configuration update. Earlier H200
heuristic JSONs are preserved under
`experiments/results/pre-three-single-pools-20260916/kda/heuristics/H200/`.
They describe old kernels and pools. Current measurements belong in
`heuristics/H200/` with their own config IDs, timings and source provenance.

## System ablations and reusable baselines

```bash
python -m experiments.kda.system.run --variant all --plan
python -m experiments.kda.system.run --variant all \
  --output experiments/results/kda/system-v1
python -m experiments.kda.tiletune.run --suite full --device hopper \
  --baseline-root experiments/results/baselines \
  --output experiments/results/kda/tiletune-revision-a
```

System runs support baseline, pipeline, grouped, multi_gpu and combined modes
on all five final FP16 cases. New TileTune output directories reuse verified baseline
bundles while the kernels, pools and measurement environment remain compatible.
Baseline XGBoost uses a fixed seed independently of TileTune repeats. Carver is
explicitly unsupported outside CUDA GEMM. See the [workflow guide](../README.md)
for GPU monitoring, baseline identity, artifact paths and arbitrary-K comparisons.
