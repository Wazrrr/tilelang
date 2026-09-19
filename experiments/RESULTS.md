# Results stored with each kernel

Each kernel owns its saved results. Begin in `experiments/<family>/results/`:
`gemm`, `grouped_gemm`, `flash_attention`, `kda`, or `gemm_fp8`. GPU-specific baseline measurements
live under `<GPU model>/baselines/` (for example, `H200/baselines/`). Generated
artifacts remain ignored by Git and excluded from source fingerprints.

The [unified memory analysis](H200_UNIFIED_MEMORY.md) on `dev-h200-new` scores
all 29,200 configs in the frozen FP16/E4M3 H200 study. Fresh IR and replay agree
for every config, and all 25 oracle winners fit strict `alpha=0.5` selection
with conservative tail ranks. The worst rank is 90/192 (46.875%). This is CPU
analysis against saved oracle timings, not a new GPU benchmark.

## Current cache-first study

The [cache-first runner](cached_study.py) fills missing measurements for the
current BF16/E4M3 contract and reuses compatible completed records. The H200 run
is stored under [h200-bf16-v2-pool7](results/studies/h200-bf16-v2-pool7/).
Its `report.md`, `comparison.json` and `comparison.csv` are generated only after
all requested comparisons finish. Progress and rejected attempts remain in the
study directory. Earlier FP16 and pre-version-2 scale-layout records are retained
as historical results and are not substituted for the current contract.

The [larger-top-K analysis](TOPK_RESULTS.md) replays the completed September 17
25-shape H200 study. At K=100, exact oracle hits are 12/25 for TileTune and 20/25
for XGBoost; at 20% of each full pool they are 12/25 and 23/25. TileTune excludes
13 oracle winners as unknown, so no larger K reaches every oracle under that
saved policy. XGBoost reaches all 25 at K=1,684 or a uniform 74% pool budget.
These results apply to the historical FP16/E4M3 study, not the incomplete newer
BF16 study.

## Baseline update rule

- Ordinary family TileTune commands only read saved baselines. They never collect, refresh
  or overwrite them, even if a matching baseline is missing.
- Call the family command with `--run-baselines` to collect or refresh brute
  force, Carver and XGBoost. This command does not run TileTune.
- A successful collection updates `baselines/current.json`. A failed refresh
  leaves that reference and the previous bundle intact.
- `baselines/runs/` retains complete prior bundles so old TileTune comparisons
  still use the exact measurements they recorded. Reuse checks the requested
  workload, dtype, GPU, configuration pool and artifact integrity. Source,
  compiler/runtime, timing and baseline-setting changes are provenance only;
  they do not require recollection. Missing cases or different pools require an
  explicit baseline invocation.

```bash
# Explicitly collect/update the GEMM baselines.
python -m experiments.gemm.tiletune.run --suite full --device hopper --run-baselines

# Read those baselines; only TileTune executes.
python -m experiments.gemm.tiletune.run --suite full --device hopper \
  --output experiments/gemm/results/tiletune/revision-a
```

The optional `--baseline-root` override retains `ROOT/TARGET/FAMILY/` storage.
Without an explicit output, family study commands create a new run directory
inside that family's `results/`. Cross-family coordinator reports can still
use a shared output; the baseline measurements remain family-owned.

## Existing records

| Family | Location | Recorded scope |
| --- | --- | --- |
| GEMM | [H200 brute-force oracle](gemm/results/H200/baselines/oracle-20260916/report.md) | Complete FP16 4096³ and 8192³ sweeps, 2,304 configs each |
| GEMM | [Best configs](gemm/heuristics/H200/README.md) | Winner summaries with hash-checked references to both full oracle tables |
| GEMM | [Published baseline bundle](gemm/results/H200/baselines/current.json) | Full-pool brute force, Carver and XGBoost; inspect the referenced completion manifest |
| GEMM | [Validation](gemm/results/H200/validation/) and [archive](gemm/results/H200/archive/) | Example comparisons, adapter checks, earlier 1,024-config results and incomplete first 2,304-config launch |
| Attention, KDA, grouped GEMM | [September 16 comparison](results/studies/all-five-h200-20260916T213019Z/PARTIAL_REPORT.md) | Historical eight-workload comparison, including model-unavailable outcomes; predates the current model and new Carver templates |
| FP8 GEMM | [Historical oracle summary](results/studies/fp8-matrix-20260917T011418Z/oracles-v2/summary.json) | 2,304 outcomes per shape under the earlier WGMMA-enabled policy; numerical and compilation failures remain failures |
| All five families | [Measured-profile GPU coverage](results/studies/workload-coverage-20260917T0306Z/summary.json) | One scored, numerically checked native configuration per final workload on H200, before the FP8 compiler-policy change; not a full-pool ranking result |
| FP8 GEMM | [Fresh MMA-policy study](results/studies/fp8-strict-mma-20260917T0320Z/RUN.md) | Fresh validation, training, full oracles and K=20 comparisons queued; consult the study's status before interpreting results |

The current FP8 experiments explicitly disable WGMMA to retain their strict
per-element accuracy contract. Earlier WGMMA-enabled measurements do not establish
performance under this compiler policy. The fresh study waits for an idle GPU;
its [queue status](results/studies/fp8-strict-mma-20260917T0320Z/queue-status.json)
and [validation status](results/studies/workload-coverage-strict-mma-20260917T0320Z/status.json)
distinguish pending work from completed results.

The GEMM oracle tables are
[4096³](gemm/results/H200/baselines/oracle-20260916/gemm_square/oracle.json) and
[8192³](gemm/results/H200/baselines/oracle-20260916/gemm_square_large/oracle.json).
Each contains 1,920 benchmarked candidates and 384 compilation failures.
Carver and XGBoost have since been collected in the published GEMM bundle.
The standalone oracle remains an oracle artifact, separate from that complete
three-baseline bundle.

Softmax is retired from the active matrix. Its historical measurements and
heuristics remain under `softmax/results/` and `softmax/heuristics/`; executable
source is archived under `results/archive/softmax-source-20260917/`.

The relocation changed storage paths only; no baseline was remeasured. Legacy
path symlinks under `experiments/results/` keep absolute references in immutable
raw records working. Mixed-family historical studies and workflow checks remain
there; the [workflow validation report](workflow_validation.md) describes their
scope. Use the family locations above for the active experiment organization.

## Interpreting saved files

`current.json` identifies a current complete baseline bundle; `complete.json`
inside that bundle verifies its identity and artifacts. Inspect its declared
pool as well: a completed subset is still a subset. `baselines.json` in a TileTune
run identifies the exact bundles it used.

A full oracle includes every declared configuration and its terminal outcome.
A ranking-only report, a top-K shortlist or a smoke check does not establish full
measured coverage. Match workload, dtype, exact configs and GPU before comparing
results. All comparison timings come from the saved oracle; provenance differences
are reported without invalidating reuse. `model_unavailable` and failed
candidates remain explicit outcomes, not successful winner timings.
