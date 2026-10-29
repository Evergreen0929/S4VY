#!/bin/bash
# Read-only preflight for canonical segmenter training/evaluation launchers.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${SEG4D_ENV:-${REPO}/env.sh}"
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
elif [ -n "${SEG4D_ENV:-}" ]; then
  echo "ERROR: SEG4D_ENV does not exist: $SEG4D_ENV" >&2
  exit 2
fi
TARGET=${1:-}
FAIL=0

need_var() {
  local n=$1 v=${!1:-}
  if [ -z "$v" ] || [[ "$v" == /path/to/* ]] || [[ "$v" == /PATH/TO/* ]] || [[ "$v" == YOUR_* ]]; then
    echo "MISSING var  $n=${v:-<unset>}"; FAIL=1
  else
    echo "OK      var  $n=$v"
  fi
}
need_file() {
  local n=$1 v=${!1:-}; need_var "$n"
  if [ -n "$v" ] && [ ! -f "$v" ]; then echo "MISSING file $n=$v"; FAIL=1; fi
}
need_dir() {
  local n=$1 v=${!1:-}; need_var "$n"
  if [ -n "$v" ] && [ ! -d "$v" ]; then echo "MISSING dir  $n=$v"; FAIL=1; fi
}
common() {
  need_var SBATCH_ACCOUNT
  need_var SBATCH_PARTITION
  need_dir SEG4D_REPO
  need_dir DATA_ROOT
  need_dir CKPT_ROOT
  need_dir VGGT_OMEGA_ROOT
  need_file VGGT_OMEGA_CKPT
  need_dir CONDA_ROOT
  need_file CONTAINER_IMAGE
}

case "$TARGET" in
  teacher) common ;;
  s4vy) common; need_file TEACHER_CKPT ;;
  s4vy_prompt) common; need_file S4VY_CKPT ;;
  s4vy_joint)
    common; need_dir PROMPT_PSEUDO_ROOT
    for subset in dl3dv matterport; do
      if [ ! -d "${PROMPT_PSEUDO_ROOT:-}/$subset" ]; then
        echo "MISSING dir  PROMPT_PSEUDO_ROOT/$subset"; FAIL=1
      else
        echo "OK      dir  PROMPT_PSEUDO_ROOT/$subset"
      fi
    done ;;
  segmenter-eval)
    need_var SBATCH_ACCOUNT; need_var SBATCH_PARTITION
    need_dir SEG4D_REPO; need_dir SEG_EVAL_ROOT; need_dir VGGT_OMEGA_ROOT; need_file VGGT_OMEGA_CKPT
    need_dir CONDA_ROOT; need_file CONTAINER_IMAGE; need_file MODEL_CKPT ;;
  *)
    echo "Usage: $0 {teacher|s4vy|s4vy_prompt|s4vy_joint|segmenter-eval}" >&2
    exit 2 ;;
esac

if [ "$FAIL" -ne 0 ]; then echo "PREFLIGHT FAILED: $TARGET"; exit 1; fi
echo "PREFLIGHT OK: $TARGET"
