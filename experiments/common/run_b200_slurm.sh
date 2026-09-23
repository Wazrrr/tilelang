#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@180

# Submit from the repository root and request 1 GPU for E1 or 4 for E2/E3.
# Usage: sbatch [site options] run_b200_slurm.sh E1 STUDY_ROOT [REUSE_MODE_ROOT]
set -euo pipefail

experiment="${1:-}"
study_root="${2:-}"
reuse_root="${3:-}"
case "${experiment}" in
    E1) expected_gpus=1 ;;
    E2|E3) expected_gpus=4 ;;
    *) echo "usage: $0 E1|E2|E3 STUDY_ROOT" >&2; exit 2 ;;
esac
if [[ -z "${study_root}" ]]; then
    echo "usage: $0 E1|E2|E3 STUDY_ROOT" >&2
    exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "this launcher must run as a Slurm batch job" >&2
    exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "Slurm did not expose CUDA_VISIBLE_DEVICES; refusing to use unallocated GPUs" >&2
    exit 2
fi

IFS=',' read -r -a visible_gpus <<<"${CUDA_VISIBLE_DEVICES}"
if [[ "${#visible_gpus[@]}" -ne "${expected_gpus}" ]]; then
    echo "${experiment} requires ${expected_gpus} visible GPU(s), got ${CUDA_VISIBLE_DEVICES}" >&2
    exit 2
fi

repo_root="${TILELANG_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "${repo_root}/experiments/common/b200.py" ]]; then
    echo "submit from the TileLang repository root or set TILELANG_REPO_ROOT" >&2
    exit 2
fi
cd "${repo_root}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

output="${study_root}/${experiment}"
python_bin="${PYTHON:-python}"
runner_pid=""
reuse_args=()
if [[ -n "${reuse_root}" ]]; then
    reuse_args=(--reuse-from "${reuse_root}")
fi

# E1/E2 are deliberately unfiltered.  Before E3 can enforce the calibrated
# family limits, prove that every exact minimum (including ties) from both
# exhaustive modes fits those limits with complete PTXAS counters.
if [[ "${experiment}" == "E3" ]]; then
    "${python_bin}" -m experiments.common.b200 \
        --audit-resource-policy "${study_root}"
fi

stop_runner() {
    if [[ -n "${runner_pid}" ]] && kill -0 "${runner_pid}" 2>/dev/null; then
        kill -TERM "${runner_pid}" 2>/dev/null || true
        set +e
        wait "${runner_pid}"
        set -e
    fi
}

requeue_before_timeout() {
    trap - USR1 TERM INT
    echo "received the Slurm wall-time warning; checkpointing the active attempt" >&2
    stop_runner
    if ! scontrol requeue "${SLURM_JOB_ID}"; then
        echo "scontrol requeue failed; the preserved output can be submitted again manually" >&2
        exit 75
    fi
    exit 0
}

terminate_for_scheduler() {
    trap - USR1 TERM INT
    echo "received a termination signal; preserving completed workloads and the partial attempt" >&2
    stop_runner
    exit 143
}

trap requeue_before_timeout USR1
trap terminate_for_scheduler TERM INT

"${python_bin}" -m experiments.common.b200 \
    --experiments "${experiment}" \
    --output "${output}" \
    "${reuse_args[@]}" \
    --resume-or-start &
runner_pid=$!
set +e
wait "${runner_pid}"
status=$?
set -e
runner_pid=""
if [[ "${status}" -ne 0 ]]; then
    exit "${status}"
fi

# E3 is dependency-chained after E1 and E2, so it can produce the shared
# comparison and oracle-retention audit once its own results are complete.
if [[ "${experiment}" == "E3" ]]; then
    "${python_bin}" -m experiments.common.b200 --combine "${study_root}"
fi
