#!/bin/bash
# Backward-compatible wrapper. Use run_libero_plus_eval.sh instead.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_libero_plus_eval.sh" "$@"
