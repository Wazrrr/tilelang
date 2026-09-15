# FlashAttention experiments

Forward attention with online softmax, in causal and noncausal modes.

## Files

```text
flash_attention/
├── README.md          Cases, knobs, commands, and results
├── cases.py           Train/validation/development/final shapes
├── spaces.py          Current/expanded/large configuration domains and rules
├── kernel.py          Kernel builder and input-generation interface
├── reference.py       Mathematical reference
├── kernels/           Implemented schedules
├── tiletune/run.py    Shared comparison protocol for this family
└── census.py          Compilation/correctness audit
```

Implementations: kernels/baseline.py, kernels/tiled.py. `kernel.make_case(workload)` supplies the suite's
builder, inputs, reference, output positions, and numerical tolerances.
Configuration generation imports only the Python standard library.

## Cases

All named-suite cases use FP16. Smoke uses the first development case.

| Split | Case | Parameters |
| --- | --- | --- |
| Training | `attention_train_a` | batch=1, heads=2, sequence=256, dim=64, causal=false |
| Training | `attention_train_b` | batch=1, heads=2, sequence=384, dim=128, causal=true |
| Validation | `attention_validation` | batch=1, heads=2, sequence=448, dim=64, causal=true |
| Development | `attention_noncausal` | batch=1, heads=4, sequence=512, dim=64, causal=false |
| Development | `attention_causal` | batch=1, heads=4, sequence=640, dim=128, causal=true |
| Final | `attention_noncausal` | batch=1, heads=4, sequence=768, dim=64, causal=false |
| Final | `attention_causal` | batch=1, heads=4, sequence=1152, dim=128, causal=true |

## Configuration space

Edit `spaces.py` to change these schedule parameters: **block_M, block_N; num_stages; threads; qk_policy, pv_policy; copy_width; implementation**.
Shapes and dtype belong in `cases.py`; they do not multiply the tuning pool.
Structural checks and proven aliases live with the family. Generic grid/hash
bookkeeping is shared through `experiments/common/`.

Current declared pools for the first final case:

| Preset | A100 | H200 |
| --- | ---: | ---: |
| current | 54 | 54 |
| expanded | 1044 | 1854 |
| large | 1024 | 1024 |
| exhaustive | 5004 | 9054 |

`large` retains the original 54 configurations and protected tiled layouts
before filling the remaining slots by deterministic parameter coverage.
`exhaustive` reproduces the old uncapped `large` domain; see the
[shared preset rules](../common/README.md#configuration-spaces).

Counts precede compilation and correctness checks. Blackwell and MI355X grids
are declared but need native device validation. Ascend needs a device manifest
with native configuration grids and an external worker. Native target support
is tracked in [the validation report](../validation.md).

## Commands

```bash
# Inspect the final shapes and complete large configuration pools, without a GPU.
python -m experiments.flash_attention.tiletune.run --suite final --device ampere --plan

# Analyze, compile, and check up to 16 smoke configurations, without latency comparison.
python -m experiments.flash_attention.tiletune.run --suite smoke --device ampere \
  --output experiments/results/flash_attention/smoke-v1

# Compare TileTune, random, XGBoost, and exhaustive search on development cases.
python -m experiments.flash_attention.tiletune.run --suite development --device ampere \
  --output experiments/results/flash_attention/development-v1

# Audit expanded development pools in resumable shards.
python -m experiments.flash_attention.census --device ampere --config-space expanded \
  --wait-idle --output experiments/results/flash_attention/census-v1

# Run final cases after the development gates pass.
python -m experiments.flash_attention.tiletune.run --suite final --device ampere \
  --development-report experiments/results/flash_attention/development-v1/acceptance.json \
  --output experiments/results/flash_attention/final-v1
```

Use `--plan` on the census to inspect its workloads, counts, and original indices.
Use `--resume` to continue completed census shards with the same inputs. The study
runner verifies existing artifacts when the same output directory is supplied.
Use a new directory after source or study-input changes.

The study uses K=20, TileTune `pipeline_time`, 10% XGBoost training/validation
samples, and seven shuffled winner checks. Final uses three seeds. The shared
[experiment overview](../README.md) documents the full protocol and output files.
A family acceptance result covers this family's requested cases and targets;
complete five-target acceptance is reported by the full matrix study.

## Existing fixed-grid and system experiments

The existing `system/run.py` commands remain available. Existing tuning commands
without named-suite options retain their original fixed-grid defaults. The old
runner is in `tiletune/legacy.py`; use `--legacy --help` to see its arguments.
See [legacy.md](legacy.md) for the complete system and fixed-grid commands.
