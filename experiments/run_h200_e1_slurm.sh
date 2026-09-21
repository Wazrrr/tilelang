#!/usr/bin/env bash
#SBATCH --job-name=h200-e1
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=512G
#SBATCH --gres=gpu:idx0:1

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: sbatch experiments/run_h200_e1_slurm.sh <new-or-resumable-result-directory>" >&2
  exit 2
fi
: "${SLURM_JOB_ID:?submit this launcher with sbatch}"

repo_root="${SLURM_SUBMIT_DIR:?Slurm did not record the submission directory}"
if [[ ! -f "${repo_root}/experiments/common/h200.py" ]]; then
  echo "submit the E1 job from the dev-h200-new repository root: ${repo_root}" >&2
  exit 2
fi
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

python - <<'PY'
import os

visible = [token for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if token]
if len(os.sched_getaffinity(0)) != 80:
    raise RuntimeError("the H200 E1 study requires an 80-CPU Slurm allocation")
if len(visible) != 1:
    raise RuntimeError(f"the H200 E1 study requires exactly one Slurm GPU, got {visible}")
print(
    {
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cpu_affinity_count": len(os.sched_getaffinity(0)),
        "cuda_visible_devices": visible,
    },
    flush=True,
)
PY

exec python -u -m experiments.common.h200 \
  --gpus 0 \
  --experiments E1 \
  --output "${result_dir}" \
  "${resume[@]}"
