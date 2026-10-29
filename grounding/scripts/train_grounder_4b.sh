#!/bin/bash
# The released 4B dual-stream grounder recipe: random new heads, four epochs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_load_env.sh"
bash "${SCRIPT_DIR}/check_paths.sh" grounder-4b
if [ "${1:-}" = "--dry-run" ]; then
  echo "DRY RUN: bash ${SCRIPT_DIR}/_submit_grounder.sh 4B"
  exit 0
fi
exec bash "${SCRIPT_DIR}/_submit_grounder.sh" 4B
