#!/usr/bin/env bash
# Configure CUDA_HOME/CXX/PYTHON in the caller's accelerator environment.
# Usage: bash experiments/portable/run_accelerator.sh [--build] --device ampere --output PATH --wait-idle
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ "${1:-}" == "--build" ]]; then
    shift
    "${CMAKE_COMMAND:-cmake}" -S . -B build
    "${CMAKE_COMMAND:-cmake}" --build build -j "${BUILD_JOBS:-16}"
fi
exec "${PYTHON:-python}" -m experiments.portable.compare "$@"
