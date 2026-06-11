#!/usr/bin/env bash
set -euo pipefail

COMMON_ENV=(
  TILELANG_DISABLE_CACHE=1
  TILELANG_AUTO_TUNING_DISABLE_CACHE=1
  TILELANG_AUTO_TUNING_CPU_UTILITIES=0.5
)

COMMON_ARGS=(
  --use_autotune
)

PROFILE_BACKENDS=(
  # event
  # cupti
  cudagraph
)

run_experiment() {
  local name="$1"
  local backend="$2"
  local example="$3"
  shift 3

  echo "===== ${name}_${backend} ====="
  env "${COMMON_ENV[@]}" python "${example}" "${COMMON_ARGS[@]}" --profile_backend "${backend}" "$@"
}

for backend in "${PROFILE_BACKENDS[@]}"; do
  run_experiment "normal_autotune" "${backend}" \
    examples/gemm/example_gemm_autotune.py

  run_experiment "all_features" "${backend}" \
    examples/gemm/example_gemm_advanced_autotune.py \
    --use_pipeline \
    --benchmark_multi_gpu \
    --benchmark_devices 0 1 \
    --enable_grouped_compile \
    --group_compile_size 2
done
