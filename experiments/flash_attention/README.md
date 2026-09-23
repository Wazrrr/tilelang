# FlashAttention experiments

The named suite calls the Blackwell-specific
[BSHD FlashAttention](../../examples/flash_attention_sm100/mha_fwd_bshd.py)
(`flashattn`) directly. Q/K/V/O use BSHD; the SM100 example uses TCGen05/TMEM
while keeping score/probability fragments and its causal loop bound.

`kernel.py` supplies input generation, the numerical reference and output
indices. The TileLang program is built directly by the example. `spaces.py`
defines the single configuration pool using the example's parameter names;
`cases.py` defines shapes. Planning imports only the Python standard library.

## Cases

All named-suite cases use BF16. Smoke uses the first development case.

| Split | Case | Parameters |
| --- | --- | --- |
| Training | `attention_train_a` | batch=1, heads=16, sequence=512, dim=64, causal=false |
| Training | `attention_train_b` | batch=2, heads=8, sequence=1024, dim=128, causal=true |
| Validation | `attention_validation` | batch=1, heads=24, sequence=2048, dim=64, causal=true |
| Development | `attention_short_causal` | batch=1, heads=32, sequence=256, dim=64, causal=true |
| Development | `attention_batched_causal` | batch=2, heads=16, sequence=1024, dim=64, causal=true |
| Development | `attention_noncausal` | batch=1, heads=32, sequence=2048, dim=128, causal=false |
| Development | `attention_causal` | batch=1, heads=32, sequence=2048, dim=128, causal=true |
| Development | `attention_long_causal` | batch=1, heads=16, sequence=4096, dim=128, causal=true |
| Final | `attention_short_causal` | batch=1, heads=32, sequence=512, dim=64, causal=true |
| Final | `attention_batched_causal` | batch=2, heads=16, sequence=2048, dim=64, causal=true |
| Final | `attention_noncausal` | batch=1, heads=32, sequence=4096, dim=128, causal=false |
| Final | `attention_causal` | batch=1, heads=32, sequence=4096, dim=128, causal=true |
| Final | `attention_long_causal` | batch=1, heads=16, sequence=8192, dim=128, causal=true |

## Configuration space

Every case uses the same complete **520-config `expanded` pool**:

| Parameter | Values |
| --- | --- |
| `block_M` | 32, 64, 128 |
| `block_N` | 16 through 256 in steps of 16 |
| `num_stages` | 0 through 12 |
| `threads` | 128 (all M), 256 (M=128) |

The example's `get_configs()` contains one config: 64/64/1/128.
Its explicit 128/128/1/128 launch is also included. The pool expands these
native tile, stage and thread parameters; it preserves the example's causal
loop, fragment recurrence, FullRow GEMMs and shared output.

Space version 11 has no alternative `current`, `large` or `exhaustive` presets
for this family. There is no cap, protected subset, target-dependent domain or
budget cap. The declared combinations are the B200 compiler-verified domain;
correctness and launch failures remain recorded.
Explicit CUDA/HIP configs must select members of this pool. Smoke and
development budgets select original indices without changing the pool.

## Commands

```bash
# Inspect final cases and complete pools without a GPU.
python -m experiments.flash_attention.tiletune.run --suite final --device hopper --plan

# Analyze, compile and check a 16-config subset.
python -m experiments.flash_attention.tiletune.run --suite smoke --device hopper \
  --output experiments/results/flash_attention/smoke-v5

# Audit the complete development pools on idle devices.
python -m experiments.flash_attention.census --device hopper --config-space expanded \
  --wait-idle --output experiments/results/flash_attention/census-v5
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
`experiments/results/pre-three-single-pools-20260916/flash_attention/heuristics/H200/`.
They describe old kernels and pools. Current measurements belong in
`heuristics/H200/` with their own config IDs, timings and source provenance.

## System ablations and reusable baselines

```bash
python -m experiments.flash_attention.system.run --variant all --plan
python -m experiments.flash_attention.system.run --variant all \
  --output experiments/results/flash_attention/system-v1
python -m experiments.flash_attention.tiletune.run --suite full --device hopper \
  --baseline-root experiments/results/baselines \
  --output experiments/results/flash_attention/tiletune-revision-a
```

System runs support baseline, pipeline, grouped, multi_gpu and combined modes
on all five final BF16 cases. New TileTune output directories reuse verified baseline
bundles while the kernels, pools and measurement environment remain compatible.
Baseline XGBoost uses a fixed seed independently of TileTune repeats. Carver is
explicitly unsupported outside CUDA GEMM. See the [workflow guide](../README.md)
for GPU monitoring, baseline identity, artifact paths and arbitrary-K comparisons.
