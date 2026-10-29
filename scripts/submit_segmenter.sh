#!/bin/bash
# Canonical segmenter submission wrapper with path preflight and --dry-run.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${SEG4D_ENV:-${REPO}/env.sh}"
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi
TARGET=${1:?Usage: $0 {teacher|s4vy|s4vy_prompt|s4vy_joint} [--dry-run]}
DRY=${2:-}
bash "${REPO}/scripts/check_repro_paths.sh" "$TARGET"
case "$TARGET" in
  teacher)       NODES=8; LAUNCHER=scripts/train_teacher.sbatch ;;
  s4vy)      NODES=8; LAUNCHER=scripts/train_s4vy.sbatch ;;
  s4vy_prompt) NODES=1; LAUNCHER=scripts/train_s4vy_prompt.sbatch ;;
  s4vy_joint)      NODES=8; LAUNCHER=scripts/train_s4vy_joint.sbatch ;;
  *) echo "ERROR: unknown target $TARGET" >&2; exit 2 ;;
esac
CMD=(sbatch --nodes="$NODES" "$LAUNCHER")
if [ "$DRY" = "--dry-run" ]; then
  printf 'DRY RUN:'; printf ' %q' "${CMD[@]}"; printf '\n'
  exit 0
fi
cd "$REPO"
"${CMD[@]}"
