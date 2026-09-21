#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: experiments/run_h200_e1.sh <new-or-resumable-result-directory>" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_dir="$1"
if [[ "${result_dir}" != /* ]]; then
  result_dir="${repo_root}/${result_dir}"
fi

resume=()
if [[ -f "${result_dir}/manifest.json" ]]; then
  resume=(--resume)
elif [[ -e "${result_dir}" ]]; then
  echo "result path exists without a resumable manifest: ${result_dir}" >&2
  exit 2
fi

source /home/ziren/anaconda3/etc/profile.d/conda.sh
conda activate tl
cd "${repo_root}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1

exec python -u -m experiments.common.h200 \
  --gpus 0 \
  --experiments E1 \
  --output "${result_dir}" \
  "${resume[@]}"
