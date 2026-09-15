# KDA experiments

The confirmed study evaluates chunk-output KDA. The family also retains recurrent KDA for separate workloads.

## Files

```text
kda/
├── README.md          Cases, knobs, commands, and results
├── cases.py           Train/validation/development/final shapes
├── spaces.py          Current/expanded/large configuration domains and rules
├── kernel.py          Kernel builder and input-generation interface
├── reference.py       Mathematical reference
├── kernels/           Implemented schedules
├── tiletune/run.py    Shared comparison protocol for this family
└── census.py          Compilation/correctness audit
```

Implementations: kernel.py (baseline), kernels/tiled.py. `kernel.make_case(workload)` supplies the suite's
builder, inputs, reference, output positions, and numerical tolerances.
Configuration generation imports only the Python standard library.

## Cases

All named-suite cases use FP16. Smoke uses the first development case.

| Split | Case | Parameters |
| --- | --- | --- |
| Training | `kda_chunk_train_a` | batch=1, heads=2, sequence=256, dim=32, value_dim=64, chunk_size=32 |
| Training | `kda_chunk_train_b` | batch=1, heads=2, sequence=256, dim=64, value_dim=96, chunk_size=64 |
| Validation | `kda_chunk_validation` | batch=1, heads=2, sequence=192, dim=48, value_dim=64, chunk_size=48 |
| Development | `kda_chunk_regular` | batch=1, heads=4, sequence=512, dim=64, value_dim=64, chunk_size=64 |
| Development | `kda_chunk_tails` | batch=1, heads=4, sequence=384, dim=48, value_dim=80, chunk_size=48 |
| Final | `kda_chunk_regular` | batch=1, heads=4, sequence=1024, dim=64, value_dim=64, chunk_size=64 |
| Final | `kda_chunk_tails` | batch=1, heads=4, sequence=768, dim=96, value_dim=80, chunk_size=48 |

## Configuration space

Edit `spaces.py` to change these schedule parameters: **block_m, block_k, block_v, block_s; stages, intra_stages; threads; implementation**.
Shapes and dtype belong in `cases.py`; they do not multiply the tuning pool.
Structural checks and proven aliases live with the family. Generic grid/hash
bookkeeping is shared through `experiments/common/`.

Current declared pools for the first final case:

| Preset | A100 | H200 |
| --- | ---: | ---: |
| current | 420 | 420 |
| expanded | 3336 | 3480 |
| large | 7384 | 7704 |

Counts precede compilation and correctness checks. Blackwell and MI355X grids
are declared but need native device validation. Ascend needs a device manifest
with native configuration grids and an external worker. Native target support
is tracked in [the validation report](../validation.md).

## Commands

```bash
# Inspect the final shapes and complete large configuration pools, without a GPU.
python -m experiments.kda.tiletune.run --suite final --device ampere --plan

# Analyze, compile, and check up to 16 smoke configurations, without latency comparison.
python -m experiments.kda.tiletune.run --suite smoke --device ampere \
  --output experiments/results/kda/smoke-v1

# Compare TileTune, random, XGBoost, and exhaustive search on development cases.
python -m experiments.kda.tiletune.run --suite development --device ampere \
  --output experiments/results/kda/development-v1

# Audit expanded development pools in resumable shards.
python -m experiments.kda.census --device ampere --config-space expanded \
  --wait-idle --output experiments/results/kda/census-v1

# Run final cases after the development gates pass.
python -m experiments.kda.tiletune.run --suite final --device ampere \
  --development-report experiments/results/kda/development-v1/acceptance.json \
  --output experiments/results/kda/final-v1
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
