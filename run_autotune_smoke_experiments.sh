#!/usr/bin/env bash
set -euo pipefail

COMMON_ENV=(
  TILELANG_DISABLE_CACHE=1
  TILELANG_AUTO_TUNING_DISABLE_CACHE=1
  TILELANG_AUTO_TUNING_CPU_UTILITIES=0.5
)

COMMON_ARGS=(
  --use_autotune
  --with_roller
  --profile_backend event
)

SMOKE_ARGS=(
  --m 4096
  --n 4096
  --k 4096
)

run_experiment() {
  local name="$1"
  local example="$2"
  shift 2

  echo "===== ${name} ====="
  env "${COMMON_ENV[@]}" python "${example}" "${COMMON_ARGS[@]}" "${SMOKE_ARGS[@]}" "$@"
}

run_experiment "normal_autotune_smoke" \
  examples/gemm/example_gemm_autotune.py

# run_experiment "normal_advanced_autotune_smoke" \
#   examples/gemm/example_gemm_advanced_autotune.py

# run_experiment "pipeline_only_smoke" \
#   examples/gemm/example_gemm_advanced_autotune.py \
#   --use_pipeline

# run_experiment "multi_gpu_2_gpus_only_smoke" \
#   examples/gemm/example_gemm_advanced_autotune.py \
#   --benchmark_multi_gpu \
#   --benchmark_devices 0 1

# run_experiment "grouped_compile_size_2_only_smoke" \
#   examples/gemm/example_gemm_advanced_autotune.py \
#   --enable_grouped_compile \
#   --group_compile_size 2
