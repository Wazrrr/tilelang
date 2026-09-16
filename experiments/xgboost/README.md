# XGBoost cost-model baseline

This is a standalone baseline using the upstream `xgboost` package. It learns
`log(latency_ms)` from workload/configuration features and selects the K smallest
predictions. It does not use TileTune's analytical scores, primitive profiles,
compiler resource observations, or pressure gates as prediction inputs.

The model is trained separately and frozen for evaluation. This implementation
is **not TVM's MetaSchedule XGBModel**: it does not use TVM's per-store feature
extractor or custom pack-sum objective. It uses standard squared-error tree
boosting on one feature row per candidate.

Training and validation each use a deterministic **10% configuration subset per
workload** by default. The subset is chosen from configuration inputs and the
seed, without looking at latencies or failures. Test workloads remain separate.

## Install

```bash
pip install -r experiments/xgboost/requirements.txt
```

The upstream CPU package is sufficient for training and prediction; accelerator
execution still uses the ordinary TileLang environment. If `xgboost` is already
installed, that package also works. Ordinary TileTune and planning commands do
not import XGBoost. No scikit-learn dependency or TVM cost-model adapter is needed.

## Collect, train, evaluate

The shared comparison command collects disjoint training/validation samples,
trains and freezes XGBoost, selects candidates, then measures the oracle:

```bash
python -m experiments.common.comparison \
  --device ampere \
  --workloads gemm_square --methods tiletune carver xgboost \
  --top-k 20 --budget-fraction 1 --xgb-sample-fraction 0.1 \
  --wait-idle --output experiments/results/xgb-comparison
```

Use a manifest for the actual device. `--plan` inspects the inputs without a GPU.
Omit `xgboost` from `--methods` when it is not wanted. Named family/suite commands
use their predefined shapes and fixed K=20 protocol. The retired family commands
accepting `--m`, `--method all` and `--xgb-model` have been removed.

For standalone training from completed collection directories:

```bash
python -m experiments.xgboost train \
  --train-runs /path/to/train-a /path/to/train-b \
  --validation-runs /path/to/validation \
  --sample-fraction 0.1 --output /path/to/model.json

python -m experiments.common.run --manifest heldout.json \
  --method xgboost --xgb-model /path/to/model.json --top-k 20
```

Collection directories contain `experiment.json`, candidate outcomes and a
completed result. The reader preserves the declared pool and original collection
cost. Sampled collections are not sampled a second time. Explicit subsets retain
their recorded size and do not establish a full-pool oracle. Models require
matching implementation, device, source and compiler identities.

For all four families, compare already saved selections with
the [offline result script](../README.md#compare-saved-selections-against-the-oracle).
It does not require the XGBoost package. To generate a new ranking from a trained
model and evaluate it against held-out measurements:

```bash
python -m experiments.xgboost evaluate --model /path/to/model.json \
  --runs /path/to/heldout --top-k 20 --output /path/to/evaluation.json
```

This predicts and freezes K before reading candidate latencies. It runs no
kernels and does not measure tuning speedup.

## Features and training rules

The feature schema includes:

- Operation, dtype, normalized mathematical parameters, and configuration knobs.
- Device kind, architecture, and observed device name.
- Kernel implementation and execution-domain identity, including source hashes,
  compiler build, available runtime versions, and measurement backend.

Numeric values become float32 columns. Categorical values use explicit one-hot
columns, with a separate unknown/missing category. Missing numeric knobs use
XGBoost's missing-value representation; new feature names absent from the
training schema are rejected. Version 2 artifacts construct that schema from
the declared, unlabeled training pools, including failed or unsampled configs.
Only the frozen subset's successful measurements contributes training labels.
This allows an implementation-specific knob to appear at inference even if its
implementation had no successful sampled measurements. Version 1 artifacts remain
readable with their original schema and domain checks; retrain for changed
kernels or additional knobs. No outcome, rank, candidate index, run name,
TileTune estimate, or compiler counter enters the features.

Training uses upstream `xgb.train`, CPU histogram trees, squared error on log
latency, and validation early stopping. Defaults adopt the four hyperparameters
reported in [WaveTune, Section 5.1](https://arxiv.org/html/2604.10187v1#S5.SS1):
600 maximum rounds (`--rounds`), depth 10 (`--max-depth`), learning rate 0.05
(`--learning-rate`), and row subsampling 0.8 (`--subsample`). With `xgb.train`,
`num_boost_round=600` corresponds to `n_estimators=600` in the sklearn interface.
Our validation patience remains 20 rounds, so the saved best iteration used for
prediction may contain fewer than 600 trees. Artifacts record the maximum,
actually trained, and best round counts. The seed remains 123 and the standalone
trainer uses four CPU threads by default.

`--subsample=0.8` draws rows from the available training data each boosting round;
`--sample-fraction=0.1` fixes which configurations supply measurements before
training. These are independent: assuming successful measurements, a 1,000-config
pool supplies 100 training rows, and each round uses approximately 80 of those.
The other 900 configurations never supply training labels. This adopts the
paper's reported hyperparameters while retaining our sampling, log-latency
objective, and validation protocol; it is not a reproduction of its full setup.

Repeated measurements of an identical candidate/context
are aggregated by median before taking the logarithm. Each training workload
receives equal total weight regardless of candidate count.

`--sample-fraction` (Python: `sample_fraction`) controls both the training and
separate validation subsets. The budget is `ceil(fraction * pool_size)`, with at
least one configuration: the default samples 231/2,304 GEMM, 32/320 attention, 72/720
chunk-KDA and 23/224 softmax configurations per workload. A seeded hash of the
canonical workload and each configuration determines selection, so changing
candidate order, timings or failure outcomes cannot bias it. Repeated runs of
one context share a single subset. Failed selections consume the sample budget
and are not replaced. Fitting requires at least two successful training labels
in total and one validation label; insufficient samples fail explicitly. A
fraction of 1 explicitly enables full-pool training. To restore the previous
training hyperparameters as well, pass `--rounds 200 --max-depth 6 --subsample 1`.

## Data and comparison contracts

- Training and validation are explicit, nonempty sets of runs completed over
  their supplied pools, including the coordinator's sampled pools.
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
measurement counts, available collection durations, and fitting time. New
artifacts also record the sampling policy, fraction, seed, original pool sizes,
selected configurations and successful selected counts for each workload. Existing
files are not overwritten. Old runs missing required source fingerprints or
compiler/runtime identity may require recollection for live evaluation.

Report offline training/validation collection and fitting costs separately from
online model loading, feature extraction, selection, compilation, and measurement.
The recorded collection durations follow the source runners' tuning-time scope;
they exclude process startup and separately prepared primitive profiles. Do not
describe a pretrained baseline as having zero training cost. Comparison winners
are assessed with the same K and the same held-out oracle grid.

Read `data.py` → `sampling.py` → `model.py`, then `../common/comparison.py` and
`../common/study.py` for collection and reuse. The TileTune core cost equations
and compiler passes are unchanged by this baseline.

Uniform sampling remains the default. Opt into implementation-stratified sampling
with `--sampling-policy implementation_stratified_config_hash_v1`. Each workload
still uses exactly `ceil(0.1 * pool_size)` attempted configurations: allocate up to
three per implementation round-robin, then distribute the remainder proportionally
to available capacity. Failed samples are not replaced. Training and validation
are sampled independently, and the policy is recorded in the artifact.
