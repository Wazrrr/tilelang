#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

export TL_TVM_FFI_TIMING_PRINT_LIMIT=0
export TL_TVM_FFI_TIMING_DETAIL=0

RESULT_DIR="benchmark/autotune/experiments_grouped_compilation/results"
mkdir -p "${RESULT_DIR}"

CPU_LIST=(1 2 4 8 16 32 64)
CPU_ARGS=()
for cpu in "${CPU_LIST[@]}"; do
  CPU_ARGS+=(--cpu-count "${cpu}")
done

COMMON_ARGS=(
  --runs 1
  --with-roller
  --topk 20
  --warmup 3
  --rep 20
  --timeout 180
  --skip-check
  --disable-cache
  --no-cache-input-tensors
  --no-detailed-measurements
  --no-pipeline
  --shape 4096x4096x4096
)

# # Baseline (group size 1 equivalent: grouped compile disabled)
# CUDA_VISIBLE_DEVICES=5 python benchmark/autotune/run_gemm_autotune_e2e.py \
#   --execution-backend auto \
#   "${CPU_ARGS[@]}" \
#   "${COMMON_ARGS[@]}" \
#   --no-grouped-compile \
#   --csv "${RESULT_DIR}/auto_group1_cpu_pow2_2_64_roller.csv"

# # Grouped compilation size 2
# CUDA_VISIBLE_DEVICES=5 python benchmark/autotune/run_gemm_autotune_e2e.py \
#   --execution-backend auto \
#   "${CPU_ARGS[@]}" \
#   "${COMMON_ARGS[@]}" \
#   --grouped-compile \
#   --group-compile-size 2 \
#   --csv "${RESULT_DIR}/auto_group2_cpu_pow2_2_64_roller.csv"

# Grouped compilation size 4
CUDA_VISIBLE_DEVICES=5 python benchmark/autotune/run_gemm_autotune_e2e.py \
  --execution-backend auto \
  "${CPU_ARGS[@]}" \
  "${COMMON_ARGS[@]}" \
  --grouped-compile \
  --group-compile-size 4 \
  --csv "${RESULT_DIR}/auto_group4_cpu_pow2_2_64_roller.csv"