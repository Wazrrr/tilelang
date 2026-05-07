# GEMM Autotune E2E Plan (Grouped Compilation Focus)

## Goal
Measure GEMM autotune end-to-end behavior across diverse `M,N,K` workloads, while only analyzing aggregate stage timings:
- `compilation_s`
- `benchmark_s`

Detailed debug timing prints and per-call adapter timing should be disabled during these experiments.

## Scope
- Script: `benchmark/autotune/run_gemm_autotune_e2e.py`
- Backend focus: `--execution-backend auto` (tvm_ffi path) and `--execution-backend cython` (nvcc path)
- CPU workers: fixed `--cpu-count 16`
- Pipeline mode: fixed `--no-pipeline`
- Grouped compilation modes:
  - baseline: `--no-grouped-compile`
  - grouped: `--grouped-compile --group-compile-size 4`

## Shape Matrix
Use a mixed set to cover different compile/runtime pressure patterns:

1. `4096x4096x4096` (balanced, large)
2. `4096x4096x512` (small-K)
3. `2048x2048x16384` (large-K)
4. `16384x1024x4096` (tall/skinny M)
5. `1024x16384x4096` (wide/skinny N)
6. `128x8192x8192` (decode-like)
7. `3584x2816x1536` (irregular)

## Global Run Settings
- `--runs 3`
- `--warmup 3`
- `--rep 20`
- `--timeout 180`
- `--without-roller`
- `--skip-check`
- `--disable-cache`
- `--no-cache-input-tensors`
- `--no-detailed-measurements`

## Disable Detailed Debug Measurements
Set these env vars before running:

```bash
export TL_TVM_FFI_TIMING_PRINT_LIMIT=0
export TL_TVM_FFI_TIMING_DETAIL=0
```

This suppresses detailed adapter timing prints so logs focus on high-level autotune progress.

## Execution Plan

### Phase 1: Baseline (no grouped compile)
Run once per backend:

```bash
python benchmark/autotune/run_gemm_autotune_e2e.py \
  --execution-backend auto \
  --cpu-count 16 \
  --runs 3 \
  --without-roller \
  --warmup 3 --rep 20 --timeout 180 \
  --skip-check --disable-cache --no-cache-input-tensors \
  --no-detailed-measurements \
  --no-pipeline \
  --no-grouped-compile \
  --shape 4096x4096x4096 \
  --shape 4096x4096x512 \
  --shape 2048x2048x16384 \
  --shape 16384x1024x4096 \
  --shape 1024x16384x4096 \
  --shape 128x8192x8192 \
  --shape 3584x2816x1536 \
  --csv benchmark/autotune/experiments_grouped_compilation/results/auto_group1.csv
```

```bash
python benchmark/autotune/run_gemm_autotune_e2e.py \
  --execution-backend cython \
  --cpu-count 16 \
  --runs 3 \
  --without-roller \
  --warmup 3 --rep 20 --timeout 180 \
  --skip-check --disable-cache --no-cache-input-tensors \
  --no-detailed-measurements \
  --no-pipeline \
  --no-grouped-compile \
  --shape 4096x4096x4096 \
  --shape 4096x4096x512 \
  --shape 2048x2048x16384 \
  --shape 16384x1024x4096 \
  --shape 1024x16384x4096 \
  --shape 128x8192x8192 \
  --shape 3584x2816x1536 \
  --csv benchmark/autotune/experiments_grouped_compilation/results/cython_group1.csv
```

### Phase 2: Grouped Compile Size 4
Run once per backend:

```bash
python benchmark/autotune/run_gemm_autotune_e2e.py \
  --execution-backend auto \
  --cpu-count 16 \
  --runs 3 \
  --without-roller \
  --warmup 3 --rep 20 --timeout 180 \
  --skip-check --disable-cache --no-cache-input-tensors \
  --no-detailed-measurements \
  --no-pipeline \
  --grouped-compile --group-compile-size 4 \
  --shape 4096x4096x4096 \
  --shape 4096x4096x512 \
  --shape 2048x2048x16384 \
  --shape 16384x1024x4096 \
  --shape 1024x16384x4096 \
  --shape 128x8192x8192 \
  --shape 3584x2816x1536 \
  --csv benchmark/autotune/experiments_grouped_compilation/results/auto_group4.csv
```

```bash
python benchmark/autotune/run_gemm_autotune_e2e.py \
  --execution-backend cython \
  --cpu-count 16 \
  --runs 3 \
  --without-roller \
  --warmup 3 --rep 20 --timeout 180 \
  --skip-check --disable-cache --no-cache-input-tensors \
  --no-detailed-measurements \
  --no-pipeline \
  --grouped-compile --group-compile-size 4 \
  --shape 4096x4096x4096 \
  --shape 4096x4096x512 \
  --shape 2048x2048x16384 \
  --shape 16384x1024x4096 \
  --shape 1024x16384x4096 \
  --shape 128x8192x8192 \
  --shape 3584x2816x1536 \
  --csv benchmark/autotune/experiments_grouped_compilation/results/cython_group4.csv
```

## Metric Policy
For analysis, only use:
- `compilation_s`
- `benchmark_s`

Ignore detailed fields (for example `compile_stage_totals_s`, `compile_stage_avg_ms`) in summary plots/tables.

## Analysis Outputs
For each `(backend, group_size, shape)`:
- mean and std of `compilation_s` across runs
- mean and std of `benchmark_s` across runs
- grouped speedup ratios vs baseline (`group4 / group1`) separately for compilation and benchmark

## Exit Criteria
- All runs finish with `status=ok`
- Collected CSVs exist for all four experiment files
- Summary clearly shows per-shape tradeoff between compile reduction and benchmark impact
