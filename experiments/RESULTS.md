# Results stored with each kernel

Each kernel owns its saved results. Begin in `experiments/<family>/results/`:
`gemm`, `grouped_gemm`, `flash_attention`, `kda`, or `gemm_fp8`. GPU-specific baseline measurements
live under `<GPU model>/baselines/` (for example, `H200/baselines/`). Generated
artifacts remain ignored by Git and excluded from source fingerprints.

The current suite has no complete oracle-retention result. See
[the memory model](H200_UNIFIED_MEMORY.md) and the active family reports.
Compilation qualification is CPU-only and is not a correctness or speed result.

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

Older non-KDA heuristic files retain their original shapes, dtypes and measured
configs. They are historical evidence, not oracles for the current pools.
Current KDA intra results are documented in [its README](kda/README.md).
