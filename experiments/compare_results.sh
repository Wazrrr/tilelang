#!/usr/bin/env bash
# Offline result comparison; PYTHON may select any Python 3.10+ interpreter.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$script_dir/..${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" -m experiments.compare_results "$@"
