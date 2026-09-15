# Softmax experiments

Full-row softmax and streamed column tiles with full-row normalization.

## Files

```text
softmax/
├── README.md          Cases, knobs, commands, and results
├── cases.py           Train/validation/development/final shapes
├── spaces.py          Current/expanded/large configuration domains and rules
├── kernel.py          Kernel builder and input-generation interface
├── reference.py       Mathematical reference
├── kernels/           Implemented schedules
├── tiletune/run.py    Shared comparison protocol for this family
└── census.py          Compilation/correctness audit
```

Implementations: kernels/baseline.py, kernels/streamed.py. `kernel.make_case(workload)` supplies the suite's
builder, inputs, reference, output positions, and numerical tolerances.
Configuration generation imports only the Python standard library.

## Cases

All named-suite cases use FP16. Smoke uses the first development case.

| Split | Case | Parameters |
| --- | --- | --- |
| Training | `softmax_train_a` | rows=256, columns=512 |
| Training | `softmax_train_b` | rows=384, columns=1536 |
| Validation | `softmax_validation` | rows=263, columns=1023 |
| Development | `softmax_aligned` | rows=512, columns=1024 |
| Development | `softmax_irregular` | rows=519, columns=769 |
| Final | `softmax_aligned` | rows=1536, columns=4096 |
| Final | `softmax_irregular` | rows=1031, columns=1537 |

## Configuration space

Edit `spaces.py` to change these schedule parameters: **block_rows, block_cols; threads; vector; row_threads; implementation**.
Shapes and dtype belong in `cases.py`; they do not multiply the tuning pool.
Structural checks and proven aliases live with the family. Generic grid/hash
bookkeeping is shared through `experiments/common/`.

Current declared pools for the first final case:

| Preset | A100 | H200 |
| --- | ---: | ---: |
| current | 6 | 6 |
| expanded | 409 | 409 |
| large | 691 | 691 |
| exhaustive | 691 | 691 |

Softmax's complete pool is already below the 1,024-config `large` cap, so its
configuration identities and order are unchanged.

Counts precede compilation and correctness checks. Blackwell and MI355X grids
are declared but need native device validation. Ascend needs a device manifest
with native configuration grids and an external worker. Native target support
is tracked in [the validation report](../validation.md).

## Commands

```bash
# Inspect the final shapes and complete large configuration pools, without a GPU.
python -m experiments.softmax.tiletune.run --suite final --device ampere --plan

# Analyze, compile, and check up to 16 smoke configurations, without latency comparison.
python -m experiments.softmax.tiletune.run --suite smoke --device ampere \
  --output experiments/results/softmax/smoke-v1

# Compare TileTune, random, XGBoost, and exhaustive search on development cases.
python -m experiments.softmax.tiletune.run --suite development --device ampere \
  --output experiments/results/softmax/development-v1

# Audit expanded development pools in resumable shards.
python -m experiments.softmax.census --device ampere --config-space expanded \
  --wait-idle --output experiments/results/softmax/census-v1

# Run final cases after the development gates pass.
python -m experiments.softmax.tiletune.run --suite final --device ampere \
  --development-report experiments/results/softmax/development-v1/acceptance.json \
  --output experiments/results/softmax/final-v1
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
