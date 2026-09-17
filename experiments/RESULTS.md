# Results stored with each kernel

Each kernel owns its saved results. Begin in `experiments/<family>/results/`:
`gemm`, `grouped_gemm`, `flash_attention`, `kda`, or `softmax`. GPU-specific baseline measurements
live under `<GPU model>/baselines/` (for example, `H200/baselines/`). Generated
artifacts remain ignored by Git and excluded from source fingerprints.

## Baseline update rule

- Ordinary TileTune runs only read saved baselines. They never collect, refresh
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
| GEMM | [Validation](gemm/results/H200/validation/) and [archive](gemm/results/H200/archive/) | Example comparisons, adapter checks, earlier 1,024-config results and incomplete first 2,304-config launch |
| Attention | [Carver adapter validation](flash_attention/results/H200/validation/carver-adapter-1789588721233592696/validation.json) | Both final 320-config pools entirely model-rejected; development shortlists all compilation-failed; no measured winner |
| Softmax | [Baseline-reuse validation](softmax/results/H200/validation/baseline-reuse-1789585935244366005/) | Two-config subset checking collection/reuse; not the full 224-config baseline |

The GEMM oracle tables are
[4096³](gemm/results/H200/baselines/oracle-20260916/gemm_square/oracle.json) and
[8192³](gemm/results/H200/baselines/oracle-20260916/gemm_square_large/oracle.json).
Each contains 1,920 benchmarked candidates and 384 compilation failures.
**Carver and XGBoost measurements for this fixed 2,304-config GEMM pool have not
been collected.** The saved standalone oracle is not a complete three-baseline
bundle and has not been relabeled as one.

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
