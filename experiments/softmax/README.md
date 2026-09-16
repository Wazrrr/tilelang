# Softmax experiments

The named suite calls [online softmax](../../examples/online_softmax/online_softmax.py) (`softmax_kernel`) directly.
The example uses two column passes and a log2/exp2 normalization recurrence. Its tail mask is shared with the experiment.

`kernel.py` supplies input generation, the numerical reference and output
indices. The TileLang program is built directly by the example. `spaces.py`
defines the single configuration pool using the example's parameter names;
`cases.py` defines shapes. Planning imports only the Python standard library.

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

Every case uses the same complete **224-config `expanded` pool**:

| Parameter | Values |
| --- | --- |
| `BLOCK_M` | 1, 2, 4, 8, 16, 32, 64 |
| `BLOCK_N` | 128, 256, 512, 1024, 2048, 4096, 8192, 16384 |
| `threads` | 64, 128, 256, 512 |

The example has no autotune grid. Its default launch (1/8192/128) is
included in this 224-config expansion. Only native row/column tiles and launch
threads vary; the example's two passes, log2/exp2 recurrence and tail mask are
preserved.

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
python -m experiments.softmax.tiletune.run --suite final --device hopper --plan

# Analyze, compile and check a 16-config subset.
python -m experiments.softmax.tiletune.run --suite smoke --device hopper \
  --output experiments/results/softmax/smoke-v5

# Audit the complete development pools on idle devices.
python -m experiments.softmax.census --device hopper --config-space expanded \
  --wait-idle --output experiments/results/softmax/census-v5
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
`experiments/results/pre-three-single-pools-20260916/softmax/heuristics/H200/`.
They describe old kernels and pools. Current measurements belong in
`heuristics/H200/` with their own config IDs, timings and source provenance.

## System ablations and reusable baselines

```bash
python -m experiments.softmax.system.run --variant all --plan
python -m experiments.softmax.system.run --variant all \
  --output experiments/results/softmax/system-v1
python -m experiments.softmax.tiletune.run --suite full --device hopper \
  --baseline-root experiments/results/baselines \
  --output experiments/results/softmax/tiletune-revision-a
```

System runs support baseline, pipeline, grouped, multi_gpu and combined modes
on both final FP16 cases. New TileTune output directories reuse verified baseline
bundles while the kernels, pools and measurement environment remain compatible.
Baseline XGBoost uses a fixed seed independently of TileTune repeats. Carver is
explicitly unsupported outside CUDA GEMM. See the [workflow guide](../README.md)
for GPU monitoring, baseline identity, artifact paths and arbitrary-K comparisons.
