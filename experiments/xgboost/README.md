# XGBoost cost-model baseline

This is a standalone baseline using the upstream `xgboost` package. It learns
`log(latency_ms)` from workload/configuration features and selects the K smallest
predictions. It does not use TileTune's analytical scores, primitive profiles,
compiler resource observations, or pressure gates as prediction inputs.

The model is trained separately and frozen for evaluation. This implementation
is **not TVM's MetaSchedule XGBModel**: it does not use TVM's per-store feature
extractor or custom pack-sum objective. It uses standard squared-error tree
boosting on one feature row per candidate.

## Install

```bash
pip install -r experiments/xgboost/requirements.txt
```

The upstream CPU package is sufficient for training and prediction; accelerator
execution still uses the ordinary TileLang environment. If `xgboost` is already
installed, that package also works. Ordinary TileTune and planning commands do
not import XGBoost. No scikit-learn dependency or TVM cost-model adapter is needed.

## Collect, train, evaluate

Use separate workload shapes for training, validation, and final evaluation.
For example, use GEMM M=256/512 for training, M=768 for validation, and M=1024 for
testing, holding the other dimensions fixed. This is an example split, not a
claim that those few shapes establish generalization.

Collect each shape with the existing exhaustive runner:

```bash
python -m experiments.gemm.tiletune.run \
  --method brute_force --m 256 --n 256 --k 256 --dtype float16 \
  --backend event --output experiments/results/xgb-data --run-name train256
```

Repeat with the other shapes and distinct run names. `--config-indices` can
explicitly restrict collection to a supplied subgrid for a smoke check. The
reader retains the supplied grid size; such a run is not a full-grid oracle.

```bash
python -m experiments.xgboost train \
  --train-runs experiments/results/xgb-data/train256 \
               experiments/results/xgb-data/train512 \
  --validation-runs experiments/results/xgb-data/validation768 \
  --output experiments/models/gemm-h200.json

python -m experiments.gemm.tiletune.run \
  --method xgboost --xgb-model experiments/models/gemm-h200.json \
  --m 1024 --n 256 --k 256 --dtype float16 --backend event --top-k 20
```

`--method all --xgb-model MODEL` includes XGBoost in the existing comparison,
alongside brute force, TileTune, and Carver where supported. Model selection
precedes the exhaustive oracle run. Winner remeasurement and Oracle@K use the
same comparison code as the existing methods. Without `--xgb-model`, `all`
preserves the existing method set.

FP8 GEMM and FlashAttention accept the same `--method xgboost --xgb-model` options
in `experiments.gemm_fp8.tiletune.run` and
`experiments.flash_attention.tiletune.run`. Supply an artifact trained for that
kernel implementation and device. XGBoost requires a finite `--top-k`.

The portable matrix supports the same baseline:

```bash
python -m experiments.portable.run --manifest heldout.json \
  --method xgboost --xgb-model experiments/models/portable-h200.json --top-k 20
```

Train its model using runs from `experiments.portable.run --method exhaustive`.
Pass individual case directories, or a parent containing only the intended
exhaustive cases, to `--train-runs` and `--validation-runs`. The portable and
dedicated runners use different kernel implementations; their model artifacts
are not interchangeable merely because both workloads are GEMMs.

For evaluation using already collected, held-out exhaustive measurements:

```bash
python -m experiments.xgboost evaluate \
  --model experiments/models/gemm-h200.json \
  --runs experiments/results/xgb-data/test1024 \
  --top-k 20 --output experiments/results/xgb-evaluation.json
```

This command predicts and freezes K before looking up those candidates' oracle
latencies. It runs no kernels and does not report an actual tuning speedup.

## Features and training rules

The feature schema includes:

- Operation, dtype, normalized mathematical parameters, and configuration knobs.
- Device kind, architecture, and observed device name.
- Kernel implementation and execution-domain identity, including source hashes,
  compiler build, available runtime versions, and measurement backend.

Numeric values become float32 columns. Categorical values use explicit one-hot
columns, with a separate unknown/missing category. Missing numeric knobs use
XGBoost's missing-value representation; new feature names absent from the
training schema are rejected. No outcome, rank, candidate index, run name,
TileTune estimate, or compiler counter enters the features.

Training uses upstream `xgb.train`, CPU histogram trees, squared error on log
latency, and validation early stopping. Defaults are 200 maximum rounds, depth 6,
learning rate 0.05, seed 123, and four CPU threads. The saved best iteration is
used for prediction. Repeated measurements of an identical candidate/context
are aggregated by median before taking the logarithm. Each training workload
receives equal total weight regardless of candidate count.

## Data and comparison contracts

- Training and validation are explicit, nonempty sets of exhaustive runs.
  Entire mathematical workloads are disjoint: changing a name, candidate subset,
  kernel implementation, or device cannot disguise overlap.
- Inference refuses workloads used for either training or validation. Test
  labels are never used for early stopping or parameter selection.
- Evaluation requires a matching device, kernel implementation/source, compiler
  build/runtime identity, and timing backend from training. This first version
  measures generalization across shapes on known execution environments; it
  makes no zero-shot operator or architecture transfer claim.
- Every imported run needs a successful completion marker and a terminal
  outcome for every supplied candidate. Only successful correctness-checked
  measurements supply latency labels. Failed candidates remain in the grid and
  failure counts; they are not assigned artificial latency labels.
- Selection uses the original supplied order to break ties. At most K candidates
  are selected before compilation. Failed selected candidates are recorded and
  never replaced, including failures while elaborating the first candidate.
  Candidate dictionaries must be unique so winners have unambiguous indices.
- The request/comparison workers verify the complete model-file SHA-256. An
  artifact cannot silently change between selection setup and worker execution.

The model JSON contains the booster, feature schema, training settings, XGBoost
version, best iteration, domain identities, split membership, run fingerprints,
measurement counts, available collection durations, and fitting time. Existing
files are not overwritten. Old runs missing required source fingerprints or
compiler/runtime identity may require recollection for live evaluation.

Report offline training/validation collection and fitting costs separately from
online model loading, feature extraction, selection, compilation, and measurement.
The recorded collection durations follow the source runners' tuning-time scope;
they exclude process startup and separately prepared primitive profiles. Do not
describe a pretrained baseline as having zero training cost. Comparison winners
are assessed with the same K and the same held-out oracle grid.

Read `data.py` → `model.py` → `integration.py` / `execution.py`. The TileTune core
cost equations and compiler passes are unchanged by this baseline.
