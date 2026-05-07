#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

export TL_TVM_FFI_TIMING_PRINT_LIMIT=0
export TL_TVM_FFI_TIMING_DETAIL=0

RESULT_DIR="benchmark/autotune/experiments_multi_gpu_benchmark/results"
mkdir -p "${RESULT_DIR}"

BENCHMARK_DEVICES_STR="${BENCHMARK_DEVICES:-0 1}"
read -r -a BENCHMARK_DEVICES <<< "${BENCHMARK_DEVICES_STR}"
BENCHMARK_DEVICE_ARGS=()
for device in "${BENCHMARK_DEVICES[@]}"; do
  BENCHMARK_DEVICE_ARGS+=(--benchmark-devices "${device}")
done

COMMON_ARGS=(
  --cpu-count 2
  --runs 1
  --with-roller
  --topk 20
  --warmup 3
  --rep 20
  --timeout 180
  --skip-check
  --disable-cache
  --no-cache-input-tensors
  --no-pipeline
  --no-grouped-compile
)

SHAPE_ARGS=(
  --shape 4096x4096x4096
  # --shape 4096x4096x512
  # --shape 2048x2048x16384
  # --shape 16384x1024x4096
  # --shape 1024x16384x4096
  # --shape 128x8192x8192
  # --shape 3584x2816x1536
)

# Baseline: single-GPU benchmark mode
CUDA_VISIBLE_DEVICES=2 python benchmark/autotune/run_gemm_autotune_e2e.py \
  --execution-backend auto \
  "${COMMON_ARGS[@]}" \
  --no-benchmark-multi-gpu \
  "${SHAPE_ARGS[@]}" \
  --csv "${RESULT_DIR}/auto_baseline_single_gpu.csv"

# Multi-GPU benchmark mode (same compile mode: no pipeline + no grouped compile)
CUDA_VISIBLE_DEVICES=2,5 python benchmark/autotune/run_gemm_autotune_e2e.py \
  --execution-backend auto \
  "${COMMON_ARGS[@]}" \
  --benchmark-multi-gpu \
  "${BENCHMARK_DEVICE_ARGS[@]}" \
  "${SHAPE_ARGS[@]}" \
  --csv "${RESULT_DIR}/auto_benchmark_multi_gpu.csv"
