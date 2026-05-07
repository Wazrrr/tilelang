#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

export TL_TVM_FFI_TIMING_PRINT_LIMIT=0
export TL_TVM_FFI_TIMING_DETAIL=0

RESULT_DIR="benchmark/autotune/experiments_pipeline_only/results"
mkdir -p "${RESULT_DIR}"

COMMON_ARGS=(
  --cpu-count 32
  --runs 1
  --without-roller
  --warmup 3
  --rep 20
  --timeout 180
  --skip-check
  --disable-cache
  --no-cache-input-tensors
  --no-detailed-measurements
  --no-grouped-compile
)

SHAPE_ARGS=(
  --shape 4096x4096x4096
  --shape 4096x4096x512
  --shape 2048x2048x16384
  --shape 16384x1024x4096
  --shape 1024x16384x4096
  --shape 128x8192x8192
  --shape 3584x2816x1536
)

# No pipeline
CUDA_VISIBLE_DEVICES=6 python benchmark/autotune/run_gemm_autotune_e2e.py \
  --execution-backend auto \
  "${COMMON_ARGS[@]}" \
  --no-pipeline \
  "${SHAPE_ARGS[@]}" \
  --csv "${RESULT_DIR}/auto_nopipeline.csv"

# CUDA_VISIBLE_DEVICES=6 python benchmark/autotune/run_gemm_autotune_e2e.py \
#   --execution-backend cython \
#   "${COMMON_ARGS[@]}" \
#   --no-pipeline \
#   "${SHAPE_ARGS[@]}" \
#   --csv "${RESULT_DIR}/cython_nopipeline.csv"

# Pipeline enabled
CUDA_VISIBLE_DEVICES=6 python benchmark/autotune/run_gemm_autotune_e2e.py \
  --execution-backend auto \
  "${COMMON_ARGS[@]}" \
  --pipeline \
  "${SHAPE_ARGS[@]}" \
  --csv "${RESULT_DIR}/auto_pipeline.csv"

# python benchmark/autotune/run_gemm_autotune_e2e.py \
#   --execution-backend cython \
#   "${COMMON_ARGS[@]}" \
#   --pipeline \
#   "${SHAPE_ARGS[@]}" \
#   --csv "${RESULT_DIR}/cython_pipeline.csv"
