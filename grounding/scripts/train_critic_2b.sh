#!/bin/bash
# Submission wrapper for the 2B critic.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_load_env.sh"
bash "${SCRIPT_DIR}/check_paths.sh" critic-2b
CMD=(sbatch --account="${GROUNDING_ACCOUNT}" --partition="${GROUNDING_TRAIN_PARTITIONS}" "${SCRIPT_DIR}/train_critic.sbatch")
if [ "${1:-}" = "--dry-run" ]; then
  printf 'DRY RUN:'; printf ' %q' "${CMD[@]}"; printf '\n'
  exit 0
fi
"${CMD[@]}"
