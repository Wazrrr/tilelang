# GEMM Autotune E2E Plan (Pipeline-Only Sweep)

## Goal
Isolate pipeline impact by changing only pipeline mode, while keeping grouped compilation disabled.

Only analyze:
- `compilation_s`
- `benchmark_s`

## Scope
- Script: `benchmark/autotune/run_gemm_autotune_e2e.py`
- Backends: `auto`, `cython`
- Fixed knobs:
  - `--cpu-count 16`
  - `--no-grouped-compile`
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

### A. Non-pipeline
```bash
python benchmark/autotune/run_gemm_autotune_e2e.py \
  --execution-backend auto \
  --cpu-count 16 \
  --runs 3 \
  --without-roller \
  --warmup 3 --rep 20 --timeout 180 \
  --skip-check --disable-cache --no-cache-input-tensors \
  --no-detailed-measurements \
  --no-grouped-compile --no-pipeline \
  --shape 4096x4096x4096 --shape 4096x4096x512 --shape 2048x2048x16384 \
  --shape 16384x1024x4096 --shape 1024x16384x4096 --shape 128x8192x8192 --shape 3584x2816x1536 \
  --csv benchmark/autotune/experiments_pipeline_only/results/auto_nopipeline.csv
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
  --no-grouped-compile --no-pipeline \
  --shape 4096x4096x4096 --shape 4096x4096x512 --shape 2048x2048x16384 \
  --shape 16384x1024x4096 --shape 1024x16384x4096 --shape 128x8192x8192 --shape 3584x2816x1536 \
  --csv benchmark/autotune/experiments_pipeline_only/results/cython_nopipeline.csv
```

### B. Pipeline enabled
```bash
python benchmark/autotune/run_gemm_autotune_e2e.py \
  --execution-backend auto \
  --cpu-count 16 \
  --runs 3 \
  --without-roller \
  --warmup 3 --rep 20 --timeout 180 \
  --skip-check --disable-cache --no-cache-input-tensors \
  --no-detailed-measurements \
  --no-grouped-compile --pipeline \
  --shape 4096x4096x4096 --shape 4096x4096x512 --shape 2048x2048x16384 \
  --shape 16384x1024x4096 --shape 1024x16384x4096 --shape 128x8192x8192 --shape 3584x2816x1536 \
  --csv benchmark/autotune/experiments_pipeline_only/results/auto_pipeline.csv
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
  --no-grouped-compile --pipeline \
  --shape 4096x4096x4096 --shape 4096x4096x512 --shape 2048x2048x16384 \
  --shape 16384x1024x4096 --shape 1024x16384x4096 --shape 128x8192x8192 --shape 3584x2816x1536 \
  --csv benchmark/autotune/experiments_pipeline_only/results/cython_pipeline.csv
```

## Analysis
- Compare `pipeline / nopipeline` for each backend and shape:
  - `compilation_s`
  - `benchmark_s`
