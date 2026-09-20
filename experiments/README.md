> The current H200 contract is [BENCHMARK_CONTRACT.md](BENCHMARK_CONTRACT.md): five families, five operations and 25 final workloads, including KDA intra-chunk. It supersedes older pool and FP8/KDA descriptions below.

# Autotuning experiments

Each kernel family owns its cases, implementations, references, configuration
spaces, and experiment commands.

The active dtype, shape, scale-layout, and architecture-specific kernel rules
are frozen in [BENCHMARK_CONTRACT.md](BENCHMARK_CONTRACT.md).

Saved measurements belong under each kernel's `results/` directory. The
[result guide](RESULTS.md) identifies the saved GEMM oracle, baseline storage,
validation records and historical data.

For unified memory scoring and strict `alpha=0.5` selection, see the
[H200 analysis and all 25 oracle scores](H200_UNIFIED_MEMORY.md). The common
runner accepts `--method top_k --metric memory --alpha 0.5`.

For a new kernel family, follow the [agent guide](.agent). For existing kernels,
start in the family folder:

| Family | Final cases | Commands and implementation |
| --- | --- | --- |
| GEMM | Five continuous-batch decode/prefill projection and FFN shapes | [gemm/](gemm/README.md) |
| FlashAttention | Five 512–8192-token causal/noncausal prefill shapes | [flash_attention/](flash_attention/README.md) |
| KDA | Token-parallel intra-chunk: five 2K–16K and batched shapes, D=128, chunk=64, sub-chunk=16 | [kda/](kda/README.md) |
| FP8 GEMM | Five block-scaled E4M3-input, BF16-output projection and FFN shapes | [gemm_fp8/](gemm_fp8/README.md) |
| Grouped GEMM | Five MoE 7168↔2048 shapes with realistic expert loads | [grouped_gemm/](grouped_gemm/README.md) |

The default matrix contains these five families, five operations and twenty-five cases. FP8 GEMM
replaces softmax; historical softmax results and its archived source remain available.

## Layout

```text
experiments/
├── gemm/                    Cases, spaces, kernels, references, commands
├── grouped_gemm/            Concatenated grouped forward GEMM study
├── flash_attention/         Same family conventions
├── kda/                     Chunk-output study
├── gemm_fp8/                Direct FP8 GEMM example study
├── common/                  Shared execution and comparison protocol
├── utils/                   Monitoring, baseline storage, result I/O and shared helpers
├── xgboost/                 Shared sampling, training, and prediction
├── suite.py                 Complete matrix or selected-family coordinator
├── manifests/               Canonical frozen study/device manifests
└── results/                 Generated artifacts, ignored by Git
```

## Fill missing records and compare all families

```bash
python -m experiments.cached_study --device hopper \
  --output experiments/results/studies/h200-bf16-v2-pool7
```

This cache-first command uses all idle, matching visible GPUs. It skips compatible
complete baseline bundles, resumes completed per-shape workers, and collects only
missing records. Run the same command to resume an interruption. Use a new output
for a later TileTune revision: valid baseline bundles remain fixed. Optional
`--families`, `--gpus`, and `--seeds` restrict the experiment; defaults cover all
five families and TileTune seeds 123, 456 and 789. `--plan` performs no GPU work.

The command collects the complete oracle, Carver's top 20, and XGBoost's top 20.
XGBoost uses two training shapes and one validation shape per family, sampling
10% of each pool. Training data, validation data, model parameters, early stopping,
validation error and fit time are saved. Final shapes are never training labels.
The immutable family bundle records every oracle outcome, each method's selected
configurations, measured winner and original tuning cost. TileTune's revision,
shape, pool, primitive profile, seed and budget identify its cached result.

Each invocation records GPU process observations. Observed contention rejects
its measurements and queues a retry on an idle GPU. Failed candidates consume
budget without replacement. The generated `report.md`, `comparison.json` and
`comparison.csv` include per-shape best configurations, Oracle@1/5/10/20/50,
actual shortlist quality, online tuning cost and reusable preparation cost.
Oracle@K uses one saved oracle table for every method; winner remeasurements
are recorded separately. Polling once per second cannot exclude shorter overlap.

The commands below remain available for explicit family collection and refresh.

## Start with one family

Planning requires only Python and does not query a GPU, compile kernels, or
create a result directory. Run commands from the repository root.

```bash
python -m experiments.gemm.tiletune.run --suite final --device hopper --plan
python -m experiments.flash_attention.tiletune.run --suite smoke --device hopper --plan
python -m experiments.kda.tiletune.run --suite development --device hopper --plan
python -m experiments.gemm_fp8.tiletune.run --suite development --device hopper --plan
python -m experiments.grouped_gemm.tiletune.run --suite development --device hopper --plan
```

A development run uses five test cases per operation, up to 256 configurations per
pool, and seed 123. Smoke uses the first case per operation and up to 16 configurations.
All five families call their example builders directly; each family README identifies its source.
Each operation has one complete `expanded` pool: GEMM 3,456, FlashAttention 576,
FP8 GEMM 576, grouped GEMM 576, and KDA intra-chunk
512 configs per case. There is no cap or structural prefilter. Final uses seeds 123, 456 and 789. All methods share the same pool for each workload. Smoke/development
budgets select indices from that pool.

```bash
python -m experiments.gemm.tiletune.run --suite development --device ampere \
  --output experiments/results/gemm/development-v1
python -m experiments.gemm.tiletune.run --suite final --device ampere \
  --development-report experiments/results/gemm/development-v1/acceptance.json \
  --output experiments/results/gemm/final-v1
```

A named-suite command compares TileTune, Carver, XGBoost, and the
exhaustive oracle. Final selection uses K=20 and `pipeline_time`; XGBoost uses
10% of each of two training pools and one validation pool per family. Its
settings are 600 rounds, depth 10, learning rate 0.05, subsampling 0.8, and
validation patience 20. Baselines use one fixed seed (123 by default) and are
read without updates across TileTune's three repeats and later revisions. Each new TileTune
winner receives seven checks. Preparation costs are recorded separately.
Carver uses existing matmul and attention templates, plus a grouped-matmul template. KDA intra-chunk currently reports
unsupported for Carver; the chunk-output template describes a different operation. Its policy equations and feasibility limits are retained.
FP8 uses the matmul template with its actual dtype. Blackwell remains unsupported
by Carver. Attention, ragged groups and KDA tails can have fully rejected pools;
these record `model_unavailable`, complete rejection reports and N/A Oracle@K.
See [model contracts](model_contracts.md) for modeling scope and limitations.

The family command checks acceptance for its requested cases and targets. Its
report identifies the scope; full five-target final acceptance requires all five
families, all five targets, and all three seeds. Final execution requires a
passing development report covering the requested cases and targets.

## Complete matrix

```bash
python -m experiments.suite --suite final --plan
python -m experiments.suite --suite development --devices ampere \
  --output experiments/results/development-v1
```

Use `--families gemm gemm_fp8` to select families or `--device-manifest FILE`
for explicit worker environments and profiles. The five targets are A100,
H200, B200/GB200, MI355X, and Ascend 910B/A2. Native implementations and calibrated
profiles still need device validation; Ascend requires supplied native grids
and its external worker. Planning a target does not establish hardware support.
The reusable baseline workflow currently executes on local CUDA devices.
Ampere, Hopper and Blackwell use the same example kernels, config pools and
commands; select `--device ampere`, `hopper` or `blackwell`. Each local target
selects a matching idle GPU within `CUDA_VISIBLE_DEVICES`, and keeps its own
baseline measurements. System ablations detect the selected GPU at runtime.
See the [CUDA portability review](cuda_portability.md) for compilation evidence
and the remaining hardware validation.
The lower-level `experiments.common.comparison` and external worker protocol
remain available for independently configured HIP/Ascend studies.

## Collect baselines explicitly, then run TileTune

```bash
# Collect brute force, Carver and XGBoost; this command does not run TileTune.
python -m experiments.gemm.tiletune.run --suite full --device hopper --run-baselines

# Ordinary runs only read the saved baseline bundle.
python -m experiments.gemm.tiletune.run --suite full --device hopper \
  --output experiments/gemm/results/tiletune/revision-a
python -m experiments.gemm.tiletune.run --suite full --device hopper \
  --output experiments/gemm/results/tiletune/revision-b
```

The shared `full` suite uses all twenty-five final cases and complete pools; a family
command uses that family's five cases, without asserting final acceptance.
Baselines are collected explicitly
per family/device/experiment identity under
`experiments/FAMILY/results/GPU/baselines/runs/`. The GPU folder uses the observed
model, for example `H200`. `baselines/current.json` identifies the current complete
bundle. The optional `--baseline-root` override keeps the custom
`ROOT/TARGET/FAMILY/` convention.
The bundle contains all oracle outcomes, Carver rankings or explicit unsupported
records, frozen XGBoost models, training/validation records and test rankings.
Its completion manifest hashes the artifacts; reuse verifies them without writing
to the bundle. Incomplete collections are archived and never treated as complete.

Reuse checks the requested workload, dtype, GPU and configuration pool. Changing
TileTune code, seeds, top K, compiler/runtime, timing settings or baseline settings
does not force recollection. The complete original source, environment and training
metadata remain recorded as provenance. Offline Oracle@K comparisons use only the
saved oracle timings, and report provenance differences without rejecting reuse.
Use `--run-baselines` explicitly when new baseline measurements are wanted.

Only `--run-baselines` collects or refreshes baseline measurements. A successful
collection updates `current.json`; a failed collection leaves the previous
reference and its measurements intact. Earlier bundles remain immutable so
previous TileTune comparisons remain reproducible. Missing or incompatible
baselines stop an ordinary TileTune invocation with an explicit rerun instruction;
they never trigger automatic collection. This storage change does not itself
rerun any baseline.

New run directories contain `baselines.json` references and per-case links to the
three saved baselines. Only TileTune executes again. Baseline collection uses
K=20 for its measured shortlist, but saves complete rankings for offline curves.
Use `--top-k` with `--suite full` for another TileTune online budget. Each case
also writes `oracle-curves.json` for K=1/5/10/20/50 and the requested K. A curve at
another K is a retrospective ranking evaluation, not a measurement of new tuning
cost. Original baseline costs are labeled as coming from the baseline bundle.

## System optimization ablations

All five family `system/run.py` entry points use the shared
[system runner](common/system.py) and the same five final cases/pools:

```bash
python -m experiments.kda.system.run --plan
python -m experiments.kda.system.run --variant all \
  --output experiments/results/kda/system-v1
```

Replace `kda` with `gemm`, `flash_attention`, `gemm_fp8` or `grouped_gemm`. `--variant all` runs
`baseline`, `pipeline`, `grouped`, `multi_gpu`, and `combined` in fresh processes
with cold caches, identical inputs and numerical checks. Pipeline overlaps
compilation/benchmarking; grouped combines compilation; multi_gpu distributes
benchmarking; combined enables all three. `--workloads` selects final case names;
`--config-indices` selects explicit original indices for development checks.
These replace the former GEMM/attention shape-specific system CLI.

By default the system runner uses all currently idle visible GPUs of one model.
`--gpus` selects physical indices; multi_gpu/combined require at least two.
System and local CUDA TileTune/baseline workers record observations every second.
A foreign compute process rejects the invocation; partial artifacts remain for
inspection and cannot complete a baseline bundle. Use a new system output or
rerun an incomplete baseline collection after the GPU becomes idle. Polling
cannot exclude overlap shorter than one second. The sharded brute-force runner
additionally retries contaminated shards automatically.

## Compare saved selections against the oracle

For a completed cross-family study, sweep larger budgets and find the exact
first-hit oracle rank for every case:

```bash
python -m experiments.topk_study --study /path/to/completed/study \
  --top-k 20 50 100 200 500 1000 --pool-percent 5 10 20 25 50 75 100
```

This writes `topk-study/report.md`, `topk.json` and `topk.csv` without GPU work.
Percentage budgets round up using the entire declared pool. Unknown/rejected
candidates remain excluded; an oracle absent from the eligible ranking is
explicitly unreachable at any K. Source hashes and the original Oracle@20 values
are checked. See the [H200 larger-top-K results](TOPK_RESULTS.md).

The offline script uses result JSONs only; it requires Python 3.10+ and no
TileLang, accelerator runtime, GPU, or XGBoost installation.

```bash
bash experiments/compare_results.sh \
  --oracle /path/to/case/brute_force/outcomes.json \
  --tiletune /path/to/case/tiletune/tiletune.json \
  --carver /path/to/case/carver/carver.json \
  --xgboost /path/to/case/xgboost/xgboost.json \
  --top-k 1 5 10 20 50 \
  --output /path/to/comparison.json
```

Omit `--xgboost` when unused. Each method is optional; supply at least one.
Run once per workload/device/seed. Input paths can also be method directories.
`--oracle` accepts an `outcomes.json` list, a `brute_force.json` report with
`configs`, an `oracle.json` object with `records`, or a heuristic JSON whose
`reference` points to the full oracle table. Heuristic references are hash
checked; their separate winner-validation timings are not mixed into the sweep.
A winner-only summary cannot evaluate unmeasured top-K alternatives.

**Oracle@K = oracle best latency / fastest successful oracle latency among the
method's first K configs.** 100% means the shortlist contains an oracle-optimal
config. The printed table shows best latency in milliseconds, Oracle@K, latency
gap, available/successful config counts and the winning oracle index. The JSON
also saves winning config dictionaries, selected indices, mapped oracle indices
and input hashes. Configuration dictionaries are matched exactly; local indices
may differ. When present, recorded workload, target, device, source and build
metadata are checked for conflicts. Bare record lists require the caller to
supply matching workloads and measurement domains.

By default, curves use the saved ranking's finite, eligible entries in saved
order. Failed compilation/checks consume K and receive no replacements. Missing
or incomplete oracle records and mismatched configs are errors. A prefix with
no successful candidate is reported as N/A. Fewer available candidates than K are explicitly
marked as a shortfall. The `saved` row separately evaluates the actual recorded
selection, including any exploration choices. Use `--order selected` to evaluate
only prefixes of `selection.selected_indices`; this cannot reconstruct larger
shortlists that were never saved. Curves are retrospective diagnostics and do
not represent new tuning or benchmarking runs.

## Configuration spaces and results

Edit mathematical shapes in each family's `cases.py` and schedule parameters
in `spaces.py`. Each family owns its structural legality and equivalence rules;
`common/spaces.py` handles deterministic enumeration and audit records. Counts
are declared candidates before compilation and correctness validation.

All five final families use only `expanded`, with the example's native parameter
names. Their full grids include the original example configs/defaults. Explicit
CUDA/HIP configs must be members of these grids. Retired vector workloads are outside this matrix. The
[configuration-space reference](common/README.md#configuration-spaces)
documents counts, failure recording and index migration.

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
[xgboost/README.md](xgboost/README.md) for the learned baseline.
Family `system/run.py` commands benchmark compiler execution strategies using
the same example builders; they are separate from tuner quality.

The retired vector experiments, compatibility modules, and repair-study
runner have been removed. The shared runner also uses the twenty-five family-owned
cases; `--smoke` chooses their development shapes.
Use the canonical `experiments.common.*` commands and `experiments.suite`.
Source fingerprints cover active code roots and exclude `results/`; historical
measurements retain their original hashes. Begin a new run after code changes.

See [workflow validation](workflow_validation.md) for the offline tests, GPU
historical checks of eight cases, and measured baseline-reuse verification.
