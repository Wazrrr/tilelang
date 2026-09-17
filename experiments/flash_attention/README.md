# FlashAttention experiments

The named suite calls [BSHD FlashAttention](../../examples/flash_attention/example_mha_fwd_bshd.py) (`flashattn`) directly.
Q/K/V/O use BSHD; the example keeps score/probability fragments and its causal loop bound.

`kernel.py` supplies input generation, the numerical reference and output
indices. The TileLang program is built directly by the example. `spaces.py`
defines the single configuration pool using the example's parameter names;
`cases.py` defines shapes. Planning imports only the Python standard library.

## Cases

All named-suite cases use FP16. Smoke uses the first development case.

| Split | Case | Parameters |
| --- | --- | --- |
| Training | `attention_train_a` | batch=1, heads=16, sequence=512, dim=128, causal=false |
| Training | `attention_train_b` | batch=1, heads=16, sequence=1024, dim=128, causal=true |
| Validation | `attention_validation` | batch=1, heads=16, sequence=2048, dim=128, causal=true |
| Development | `attention_short_causal` | batch=1, heads=32, sequence=256, dim=128, causal=true |
| Development | `attention_medium_causal` | batch=1, heads=32, sequence=1024, dim=128, causal=true |
| Development | `attention_noncausal` | batch=1, heads=32, sequence=2048, dim=128, causal=false |
| Development | `attention_causal` | batch=1, heads=32, sequence=2048, dim=128, causal=true |
| Development | `attention_long_causal` | batch=1, heads=32, sequence=4096, dim=128, causal=true |
| Final | `attention_short_causal` | batch=1, heads=32, sequence=512, dim=128, causal=true |
| Final | `attention_medium_causal` | batch=1, heads=32, sequence=2048, dim=128, causal=true |
| Final | `attention_noncausal` | batch=1, heads=32, sequence=4096, dim=128, causal=false |
| Final | `attention_causal` | batch=1, heads=32, sequence=4096, dim=128, causal=true |
| Final | `attention_long_causal` | batch=1, heads=32, sequence=8192, dim=128, causal=true |

## Configuration space

Every case uses the same complete **320-config `expanded` pool**:

| Parameter | Values |
| --- | --- |
| `block_M` | 32, 64, 128, 192, 256 |
| `block_N` | 16, 32, 48, 64, 96, 128, 192, 256 |
| `num_stages` | 0, 1, 2, 3 |
| `threads` | 128, 256 |

The example's `get_configs()` contains one config: 64/64/1/128.
Its explicit 128/128/1/128 launch is also included. The pool expands these
native tile, stage and thread parameters; it preserves the example's causal
loop, fragment recurrence, FullRow GEMMs and shared output.

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
python -m experiments.flash_attention.tiletune.run --suite final --device hopper --plan

# Analyze, compile and check a 16-config subset.
python -m experiments.flash_attention.tiletune.run --suite smoke --device hopper \
  --output experiments/flash_attention/results/smoke-v5

# Audit the complete development pools on idle devices.
python -m experiments.flash_attention.census --device hopper --config-space expanded \
  --wait-idle --output experiments/flash_attention/results/census-v5
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
  --output experiments/flash_attention/results/system-v1
python -m experiments.flash_attention.tiletune.run --suite full --device hopper --run-baselines
python -m experiments.flash_attention.tiletune.run --suite full --device hopper \
  --output experiments/flash_attention/results/tiletune-revision-a
```

System runs support baseline, pipeline, grouped, multi_gpu and combined modes
on all five final FP16 cases. New TileTune output directories reuse verified baseline
bundles while the kernels, pools and measurement environment remain compatible.
Baselines live under `results/<GPU model>/baselines/`, with `current.json` pointing
to the saved bundle. Only `--run-baselines` collects or refreshes them; ordinary
TileTune runs are read-only and require an existing compatible bundle.
Baseline XGBoost uses a fixed seed independently of TileTune repeats. See the [workflow guide](../README.md)
for GPU monitoring, baseline identity, artifact paths and arbitrary-K comparisons.

## Carver baseline

`carver.py` adapts the repository's original `FlashAttentionTemplate` and
`TensorCorePolicy` to all 320 configs for CUDA FP16/BF16 attention. Nothing under
`tilelang/carver/` is modified. Selected configs execute the same BSHD example
kernel, including softmax and causal behavior, used by every other method.

| Example parameter | Original policy input |
| --- | --- |
| `block_M` | PV output tile `[1, block_M, head_dim]` |
| `block_N` | PV reduction step; QK uses Carver's default reduction step |
| `num_stages` | Existing `pipeline_stage`, with 0 and 1 mapped to 1 |
| `threads` | Original block-size feasibility check for both GEMM nodes |

The score is exactly `(traffic_bytes + 1) * num_wave`. All configs retain their
original indices; ties retain pool order. The adapter neither expands tiles nor
replaces rejected candidates. Each record includes both node tiles, reduction
steps, memory estimates, score and rejection reason.

Carver's graph has two connected GEMMs, omits softmax/scaling and causal work
reduction, and models transposed second operands. It propagates a full
sequence-wide QK tile instead of the example's streamed KV tiles. These model
limitations are preserved, not repaired. `sm_90a` is normalized to `sm_90` only
for Carver; compilation keeps the actual target.

On H200, the unchanged policy's 49,152-byte shared-memory limit rejects every
config in both final pools: the minimum estimates are 52,224 bytes (noncausal)
and 78,848 bytes (causal). Such a run saves all 320 rejections and an empty
selection, returns `model_unavailable`, and reports N/A Oracle@K. It is a recorded
baseline outcome that can be cached and reused. Smaller development cases
selected 16 noncausal and 4 causal configs at K=20; all failed compilation in
the unchanged example (fragment-layout/FullRow warp-partition errors). Their
`failed` results and per-config errors are retained without replacements. No
attention winner latency was measured in this adapter validation; see the
[validation record](../workflow_validation.md#attention-carver-adapter).

```bash
python -m experiments.common.run --devices hopper --method carver \
  --workloads attention_noncausal attention_causal --top-k 20 \
  --output experiments/flash_attention/results/carver-final
# Add --smoke to run the two development shapes with the same 320-config pool.
```
