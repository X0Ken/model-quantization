#!/usr/bin/env bash
# Adapted from Chiro's entry point: explicit paths, no deletion, DP8 by default.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON:-python3}" "$HERE/run_stage1.py" "$@"
