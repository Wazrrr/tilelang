# GEMM Autotune E2E Plan (Pipeline + Grouped Compilation)

## Goal
Evaluate combined effects when grouped compilation is enabled and pipeline mode is toggled.

Only analyze:
- `compilation_s`
- `benchmark_s`

## Scope
- Script: `benchmark/autotune/run_gemm_autotune_e2e.py`
- Backends: `auto`, `cython`
- Fixed knobs:
  - `--cpu-count 16`
  - `--grouped-compile --group-compile-size 4`
  - `--without-roller`
  - `--skip-check --disable-cache --no-cache-input-tensors`
  - `--no-detailed-measurements`
  - `--runs 3 --warmup 3 --rep 20 --timeout 180`

## Shapes
- `4096x4096x4096`
- `4096x4096x512`
- `2048x2048x16384`
- `16384x1024x4096`
- `1024x16384x4096`
- `128x8192x8192`
- `3584x2816x1536`

## Disable Detailed Debug Measurements
```bash
export TL_TVM_FFI_TIMING_PRINT_LIMIT=0
export TL_TVM_FFI_TIMING_DETAIL=0
```

## Runs

### A. Grouped + non-pipeline
```bash
python benchmark/autotune/run_gemm_autotune_e2e.py \
  --execution-backend auto \
  --cpu-count 16 \
  --runs 3 \
  --without-roller \
  --warmup 3 --rep 20 --timeout 180 \
  --skip-check --disable-cache --no-cache-input-tensors \
  --no-detailed-measurements \
  --grouped-compile --group-compile-size 4 --no-pipeline \
  --shape 4096x4096x4096 --shape 4096x4096x512 --shape 2048x2048x16384 \
  --shape 16384x1024x4096 --shape 1024x16384x4096 --shape 128x8192x8192 --shape 3584x2816x1536 \
  --csv benchmark/autotune/experiments_pipeline_grouped/results/auto_group4_nopipeline.csv
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
  --grouped-compile --group-compile-size 4 --no-pipeline \
  --shape 4096x4096x4096 --shape 4096x4096x512 --shape 2048x2048x16384 \
  --shape 16384x1024x4096 --shape 1024x16384x4096 --shape 128x8192x8192 --shape 3584x2816x1536 \
  --csv benchmark/autotune/experiments_pipeline_grouped/results/cython_group4_nopipeline.csv
```

### B. Grouped + pipeline
```bash
python benchmark/autotune/run_gemm_autotune_e2e.py \
  --execution-backend auto \
  --cpu-count 16 \
  --runs 3 \
  --without-roller \
  --warmup 3 --rep 20 --timeout 180 \
  --skip-check --disable-cache --no-cache-input-tensors \
  --no-detailed-measurements \
  --grouped-compile --group-compile-size 4 --pipeline \
  --shape 4096x4096x4096 --shape 4096x4096x512 --shape 2048x2048x16384 \
  --shape 16384x1024x4096 --shape 1024x16384x4096 --shape 128x8192x8192 --shape 3584x2816x1536 \
  --csv benchmark/autotune/experiments_pipeline_grouped/results/auto_group4_pipeline.csv
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
  --grouped-compile --group-compile-size 4 --pipeline \
  --shape 4096x4096x4096 --shape 4096x4096x512 --shape 2048x2048x16384 \
  --shape 16384x1024x4096 --shape 1024x16384x4096 --shape 128x8192x8192 --shape 3584x2816x1536 \
  --csv benchmark/autotune/experiments_pipeline_grouped/results/cython_group4_pipeline.csv
```

## Analysis
- Compare `group4_pipeline / group4_nopipeline` for each backend and shape:
  - `compilation_s`
  - `benchmark_s`
