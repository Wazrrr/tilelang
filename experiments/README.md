# Autotuning experiments

Each kernel family owns its cases, implementations, references, configuration
spaces, and experiment commands. Start in the family folder:

| Family | Final FP16 cases | Commands and implementation |
| --- | --- | --- |
| GEMM | 4096³ and 8192³ | [gemm/](gemm/README.md) |
| FlashAttention | Noncausal and causal | [flash_attention/](flash_attention/README.md) |
| KDA | Chunk output with equal and unequal head dimensions | [kda/](kda/README.md) |
| Softmax | Aligned and irregular rows | [softmax/](softmax/README.md) |

## Layout

```text
experiments/
├── gemm/                    Cases, spaces, kernels, references, commands
├── flash_attention/         Same family conventions
├── kda/                     Chunk-output study and recurrent implementation
├── softmax/                 Full-row and streamed softmax
├── vector/                  Other normalization/reduction/elementwise kernels
├── gemm_fp8/                Existing FP8 fixed-grid and system experiments
├── common/                  Shared execution and comparison protocol
├── xgboost/                 Shared sampling, training, and prediction
├── suite.py                 Complete matrix or selected-family coordinator
├── manifests/               Canonical frozen study/device manifests
├── portable/                Compatibility entry points
└── results/                 Generated artifacts, ignored by Git
```

## Start with one family

Planning requires only Python and does not query a GPU, compile kernels, or
create a result directory. Run commands from the repository root.

```bash
python -m experiments.gemm.tiletune.run --suite final --device ampere --plan
python -m experiments.flash_attention.tiletune.run --suite smoke --device ampere --plan
python -m experiments.kda.tiletune.run --suite development --device ampere --plan
python -m experiments.softmax.tiletune.run --suite development --device ampere --plan
```

A development run uses two test cases per family, up to 256 configurations per
pool, and seed 123. Smoke uses the first case and up to 16 configurations.
Final uses the full `large` pools and seeds 123, 456, and 789.

```bash
python -m experiments.gemm.tiletune.run --suite development --device ampere \
  --output experiments/results/gemm/development-v1
python -m experiments.gemm.tiletune.run --suite final --device ampere \
  --development-report experiments/results/gemm/development-v1/acceptance.json \
  --output experiments/results/gemm/final-v1
```

A named-suite command compares TileTune, random selection, XGBoost, and the
exhaustive oracle. Selection uses fixed K=20 and `pipeline_time`; XGBoost uses
10% of each of two training pools and one validation pool per family. Its
settings are 600 rounds, depth 10, learning rate 0.05, subsampling 0.8, and
validation patience 20. All seeds' selections finish before test oracles;
winners receive seven shuffled checks. Preparation costs are recorded separately.

The family command checks acceptance for its requested cases and targets. Its
report identifies the scope; full five-target final acceptance requires all four
families, all five targets, and all three seeds. Final execution requires a
passing development report covering the requested cases and targets.

## Complete matrix

```bash
python -m experiments.suite --suite final --plan
python -m experiments.suite --suite development --devices ampere \
  --output experiments/results/development-v1
```

Use `--families gemm softmax` to select families or `--device-manifest FILE`
for explicit worker environments and profiles. The five targets are A100,
H200, B200/GB200, MI355X, and Ascend 910B/A2. Native implementations and calibrated
profiles still need device validation; Ascend requires supplied native grids
and its external worker. Planning a target does not establish hardware support.

## Configuration spaces and results

Edit mathematical shapes in each family's `cases.py` and schedule parameters
in `spaces.py`. Each family owns its structural legality and equivalence rules;
`common/spaces.py` handles deterministic enumeration and audit records. Counts
are declared candidates before compilation and correctness validation.

The final manifest at [manifests/five_target_final.json](manifests/five_target_final.json)
is a frozen snapshot of the family definitions. Final planning checks they match.
Source, profiles, settings, and configuration subsets are recorded in
`study-lock.json`. Use a new output directory after changing study inputs.

| Artifact | Meaning |
| --- | --- |
| `study-lock.json` | Frozen study inputs and source/profile hashes |
| `smoke.json` | Analysis, compilation, correctness, instruction evidence |
| `SEED/TARGET/comparison.json` | Method results, ranking quality, winner checks |
| `oracle/` | Shared exhaustive measurements |
| `acceptance.json` | Per-case and per-seed gates, costs, and study scope |

See [common/README.md](common/README.md) for worker and diagnostic details,
[xgboost/README.md](xgboost/README.md) for the learned baseline, and
[legacy.md](legacy.md) for existing fixed-grid/system experiments.
Old `experiments.portable` commands and imports remain compatible.
