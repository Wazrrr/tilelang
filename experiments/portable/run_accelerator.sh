#!/usr/bin/env bash
# Compatibility entry point for the shared accelerator launcher.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../common/run_accelerator.sh" "$@"
