# GEMM experiments

Tiled matrix multiplication with FP32 accumulation. The advanced-autotune implementation also has a shared-memory epilogue.

## Files

```text
gemm/
├── README.md          Cases, knobs, commands, and results
├── cases.py           Train/validation/development/final shapes
├── spaces.py          Current/expanded/large configuration domains and rules
├── kernel.py          Kernel builder and input-generation interface
├── reference.py       Mathematical reference
├── kernels/           Implemented schedules
├── tiletune/run.py    Shared comparison protocol for this family
└── census.py          Compilation/correctness audit
```

Implementations: kernels/advanced.py, kernels/tiled.py. `kernel.make_case(workload)` supplies the suite's
builder, inputs, reference, output positions, and numerical tolerances.
Configuration generation imports only the Python standard library.

## Cases

All named-suite cases use FP16. Smoke uses the first development case.

| Split | Case | Parameters |
| --- | --- | --- |
| Training | `gemm_train_a` | m=512, n=512, k=512 |
| Training | `gemm_train_b` | m=1024, n=256, k=768 |
| Validation | `gemm_validation` | m=384, n=768, k=512 |
| Development | `gemm_square` | m=1024, n=1024, k=1024 |
| Development | `gemm_square_large` | m=2048, n=2048, k=2048 |
| Final | `gemm_square` | m=4096, n=4096, k=4096 |
| Final | `gemm_square_large` | m=8192, n=8192, k=8192 |

## Configuration space

Edit `spaces.py` to change these schedule parameters: **block_m, block_n, block_k; stages; threads; warp_policy; swizzle_panel**.
Shapes and dtype belong in `cases.py`; they do not multiply the tuning pool.
Structural checks and proven aliases live with the family. Generic grid/hash
bookkeeping is shared through `experiments/common/`.

Current declared pools for the first final case:

| Preset | A100 | H200 |
| --- | ---: | ---: |
| current | 108 | 108 |
| expanded | 2060 | 3000 |
| large | 1024 | 1024 |
| exhaustive | 6180 | 9000 |

`large` retains the original 108 configurations, protected central and
rectangular tile neighborhoods, and deterministic parameter coverage. Both
archived A100 GEMM sweep winners remain in the pool. `exhaustive` reproduces the
old uncapped `large` domain; see the [shared preset rules](../common/README.md#configuration-spaces).

Counts precede compilation and correctness checks. Blackwell and MI355X grids
are declared but need native device validation. Ascend needs a device manifest
with native configuration grids and an external worker. Native target support
is tracked in [the validation report](../validation.md).

## Commands

```bash
# Inspect the final shapes and complete large configuration pools, without a GPU.
python -m experiments.gemm.tiletune.run --suite final --device ampere --plan

# Analyze, compile, and check up to 16 smoke configurations, without latency comparison.
python -m experiments.gemm.tiletune.run --suite smoke --device ampere \
  --output experiments/results/gemm/smoke-v1

# Compare TileTune, random, XGBoost, and exhaustive search on development cases.
python -m experiments.gemm.tiletune.run --suite development --device ampere \
  --output experiments/results/gemm/development-v1

# Audit expanded development pools in resumable shards.
python -m experiments.gemm.census --device ampere --config-space expanded \
  --wait-idle --output experiments/results/gemm/census-v1

# Run final cases after the development gates pass.
python -m experiments.gemm.tiletune.run --suite final --device ampere \
  --development-report experiments/results/gemm/development-v1/acceptance.json \
  --output experiments/results/gemm/final-v1
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

The advanced implementation uses `spaces.advanced_configurations()` (288 choices),
with its original `block_M`/`thread_num` argument names. The suite tiled
implementation uses `current`, `expanded`, and `large`, with `block_m`/`threads`.
The grids are specific to their implementations. Carver remains available in
the legacy comparison for supported GEMM configurations.
