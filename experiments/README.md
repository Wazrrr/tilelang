# Autotuning experiments

Each kernel family owns its cases, implementations, references, configuration
spaces, and experiment commands.

For a new kernel family, follow the [agent guide](.agent). For existing kernels,
start in the family folder:

| Family | Final cases | Commands and implementation |
| --- | --- | --- |
| GEMM | Five continuous-batch decode/prefill projection and FFN shapes | [gemm/](gemm/README.md) |
| FlashAttention | Five 512–8192-token causal/noncausal prefill shapes | [flash_attention/](flash_attention/README.md) |
| KDA | Five 2K–16K and batched chunk-output shapes at DK=DV=128 | [kda/](kda/README.md) |
| FP8 GEMM | Five E4M3 decode/prefill projection and FFN shapes | [gemm_fp8/](gemm_fp8/README.md) |
| Grouped GEMM | Five MoE 7168↔2048 shapes with realistic expert loads | [grouped_gemm/](grouped_gemm/README.md) |

Grouped GEMM follows the same family structure with a 192-config pool. The
default matrix has five families and twenty-five cases. Its five holdouts have a
separate frozen manifest, and `--families` can still select any subset.

## Layout

```text
experiments/
├── gemm/                    Cases, spaces, kernels, references, commands
├── gemm_fp8/                Native E4M3 GEMM study
├── grouped_gemm/            Opt-in concatenated grouped forward GEMM study
├── flash_attention/         Same family conventions
├── kda/                     Direct chunk-output example study
├── common/                  Shared execution and comparison protocol
├── utils/                   Monitoring, baseline storage, result I/O and shared helpers
├── xgboost/                 Shared sampling, training, and prediction
├── suite.py                 Complete matrix or selected-family coordinator
├── manifests/               Canonical frozen study/device manifests
└── results/                 Generated artifacts, ignored by Git
```

## Start with one family

Planning requires only Python and does not query a GPU, compile kernels, or
create a result directory. Run commands from the repository root.

```bash
python -m experiments.gemm.tiletune.run --suite final --device ampere --plan
python -m experiments.flash_attention.tiletune.run --suite smoke --device ampere --plan
python -m experiments.kda.tiletune.run --suite development --device ampere --plan
python -m experiments.gemm_fp8.tiletune.run --suite development --device blackwell --plan
```

A development run uses five test cases per family, up to 256 configurations per
pool, and seed 123. Smoke uses the first case and up to 16 configurations.
The four final kernels call their [example builders directly](example_alignment.md).
Each family has one complete `expanded` pool: GEMM 2,304, FlashAttention 320,
KDA 720, FP8 GEMM 2,304, and grouped GEMM 192 configs per case. There is no cap or structural
prefilter. Final uses seeds 123, 456 and 789. All methods share the same pool for each workload. Smoke/development
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
reused across TileTune's three repeats and later revisions. Each new TileTune
winner receives seven checks. Preparation costs are recorded separately.
Carver resolves every workload through one canonical template selector:
`MatmulTemplate` for GEMM, `FP8MatmulTemplate` for FP8 GEMM,
`FlashAttentionTemplate` for attention, `GroupedMatmulTemplate` for grouped
GEMM, and `KDAChunkTemplate` for KDA. Every adapter scores the exact shared
experiment pool rather than a separately generated grid. FP8 templates retain
the kernel format while lowering `float8_e4m3fn` to Carver's tensorizable
`float8_e4m3` spelling internally.

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

Use `--families gemm gemm_fp8` to select families or `--device-manifest FILE`
for explicit worker environments and profiles. The five targets are A100,
H200, B200/GB200, MI355X, and Ascend 910B/A2. Native implementations and calibrated
profiles still need device validation; Ascend requires supplied native grids
and its external worker. Planning a target does not establish hardware support.
The reusable baseline workflow currently executes on local CUDA devices.
The lower-level `experiments.common.comparison` and external worker protocol
remain available for independently configured HIP/Ascend studies.

## Reuse baselines across TileTune revisions

```bash
python -m experiments.suite --suite full --devices hopper \
  --baseline-root experiments/results/baselines \
  --output experiments/results/tiletune/revision-a

# After changing TileTune, use a new run directory and the same baseline root.
python -m experiments.suite --suite full --devices hopper \
  --baseline-root experiments/results/baselines \
  --output experiments/results/tiletune/revision-b
```

`full` uses all twenty-five final cases and complete pools without asserting final
acceptance. Family commands support the same flags. Baselines are collected once
per family/device/experiment identity under `baseline-root/TARGET/FAMILY/HASH/`.
The bundle contains all oracle outcomes, Carver rankings or explicit unsupported
records, frozen XGBoost models, training/validation records and test rankings.
Its completion manifest hashes the artifacts; reuse verifies them without writing
to the bundle. Incomplete collections are archived and never treated as complete.

The identity covers example kernels, ordered pools, shapes/dtypes, compiler
sources/toolchain, runtime/driver, GPU model and measurement settings. Baseline
implementations, XGBoost settings and `--baseline-seed` also identify the bundle.
Changes confined to `tilelang/tiletune/` or `tiletune_core/`, TileTune repeat seeds,
and evaluation K reuse the same baselines. Compiler/JIT/profiler or kernel changes
select a new bundle. The kernel compiler must be rebuilt after native source
changes; source identity does not rebuild the installed compiler.

New run directories contain `baselines.json` references and per-case links to the
three saved baselines. Only TileTune executes again. Baseline collection uses
K=20 for its measured shortlist, but saves complete rankings for offline curves.
Use `--top-k` with `--suite full` for another TileTune online budget. Each case
also writes `oracle-curves.json` for K=1/5/10/20/50 and the requested K. A curve at
another K is a retrospective ranking evaluation, not a measurement of new tuning
cost. Original baseline costs are labeled as coming from the baseline bundle.

## System optimization ablations

All four default-family `system/run.py` entry points use the shared
[system runner](common/system.py) and the same five final cases/pools:

```bash
python -m experiments.kda.system.run --plan
python -m experiments.kda.system.run --variant all \
  --output experiments/results/kda/system-v1
```

Replace `kda` with `gemm`, `flash_attention` or `gemm_fp8`. `--variant all` runs
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

All four final families use only `expanded`, with the example's native parameter
names. Their full grids include the original example configs/defaults. Explicit
CUDA/HIP configs must be members of these grids. Separate vector workloads
retain their existing presets. The
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

The retired FP8/vector experiments, compatibility modules, and repair-study
runner have been removed. The shared runner also uses the twenty family-owned
cases; `--smoke` chooses their development shapes.
Use the canonical `experiments.common.*` commands and `experiments.suite`.
Source fingerprints cover active code roots and exclude `results/`; historical
measurements retain their original hashes. Begin a new run after code changes.

See [workflow validation](workflow_validation.md) for the offline tests, GPU
checks of the earlier eight-case matrix, and measured baseline-reuse verification.
