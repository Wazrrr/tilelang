#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-/tmp/tilelang_autotune_filter_experiments}"
QUALITY_ACTION="${QUALITY_ACTION:-reject}"
GEMM_M="${GEMM_M:-4096}"
GEMM_N="${GEMM_N:-4096}"
GEMM_K="${GEMM_K:-4096}"
ATTN_BATCH="${ATTN_BATCH:-1}"
ATTN_HEADS="${ATTN_HEADS:-16}"
ATTN_SEQ_Q="${ATTN_SEQ_Q:-1024}"
ATTN_SEQ_KV="${ATTN_SEQ_KV:-1024}"
ATTN_DIM="${ATTN_DIM:-64}"
WARMUP="${WARMUP:-1}"
REP="${REP:-1}"
TIMEOUT="${TIMEOUT:-180}"
GROUP_COMPILE_SIZE="${GROUP_COMPILE_SIZE:-8}"
TILELANG_AUTO_TUNING_CPU_UTILITIES="${TILELANG_AUTO_TUNING_CPU_UTILITIES:-0.5}"

export TILELANG_DISABLE_CACHE="${TILELANG_DISABLE_CACHE:-1}"
export TILELANG_AUTO_TUNING_DISABLE_CACHE="${TILELANG_AUTO_TUNING_DISABLE_CACHE:-1}"
export TILELANG_AUTO_TUNING_CPU_UTILITIES

max_config_args=()
if [[ -n "${MAX_CONFIGS:-}" ]]; then
  max_config_args=(--max-configs "${MAX_CONFIGS}")
fi

mkdir -p "${OUT_ROOT}"

python examples/gemm/run_gemm_family_filter_experiment.py \
  --family all \
  --m "${GEMM_M}" \
  --n "${GEMM_N}" \
  --k "${GEMM_K}" \
  --warmup "${WARMUP}" \
  --rep "${REP}" \
  --timeout "${TIMEOUT}" \
  --out-root "${OUT_ROOT}/gemm" \
  --quality-action "${QUALITY_ACTION}" \
  --grouped \
  --group-compile-size "${GROUP_COMPILE_SIZE}" \
  --disable-cache \
  "${max_config_args[@]}"

python examples/flash_attention/run_attention_quality_survey.py \
  --batch "${ATTN_BATCH}" \
  --heads "${ATTN_HEADS}" \
  --seq-q "${ATTN_SEQ_Q}" \
  --seq-kv "${ATTN_SEQ_KV}" \
  --dim "${ATTN_DIM}" \
  --warmup "${WARMUP}" \
  --rep "${REP}" \
  --out-dir "${OUT_ROOT}/attention" \
  --quality-action "${QUALITY_ACTION}" \
  --group-compile-size "${GROUP_COMPILE_SIZE}" \
  "${max_config_args[@]}"

echo "summary=${OUT_ROOT}"
