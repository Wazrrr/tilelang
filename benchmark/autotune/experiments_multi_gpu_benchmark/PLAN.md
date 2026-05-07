# GEMM Autotune E2E Plan (Benchmark Multi-GPU Focus)

## Goal
Isolate benchmark-stage speedup from multi-GPU benchmarking while keeping compile behavior fixed.

Primary metrics:
- `compilation_s`
- `benchmark_s`

## Scope
- Script: `benchmark/autotune/run_gemm_autotune_e2e.py`
- Backend: `--execution-backend auto`
- Fixed settings:
  - `--no-pipeline`
  - `--no-grouped-compile`
  - `--without-roller`
  - `--skip-check --disable-cache --no-cache-input-tensors`
  - `--no-detailed-measurements`
  - `--runs 1 --warmup 3 --rep 20 --timeout 180`
  - `--cpu-count 16`

## Comparison
1. Baseline single-GPU benchmark:
   - `--no-benchmark-multi-gpu`
2. Multi-GPU benchmark:
   - `--benchmark-multi-gpu`
   - `--benchmark-devices ...`

Both runs use identical shapes and compile-mode knobs so deltas are attributable to benchmark parallelism.

## Shapes
- `4096x4096x4096`
- `4096x4096x512`
- `2048x2048x16384`
- `16384x1024x4096`
- `1024x16384x4096`
- `128x8192x8192`
- `3584x2816x1536`

## Run Script
Use:

```bash
benchmark/autotune/experiments_multi_gpu_benchmark/bash.sh
```

Optional device override:

```bash
BENCHMARK_DEVICES="0 1 2 3" benchmark/autotune/experiments_multi_gpu_benchmark/bash.sh
```
