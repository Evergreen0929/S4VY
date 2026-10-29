#!/bin/bash
# Read-only path preflight. Usage: bash scripts/check_paths.sh <target>
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_load_env.sh"
TARGET=${1:-all}
FAIL=0

need_var() {
  local n=$1 v=${!1:-}
  if [ -z "$v" ] || [[ "$v" == /path/to/* ]] || [[ "$v" == /PATH/TO/* ]] || [[ "$v" == YOUR_* ]]; then
    echo "MISSING var  $n=${v:-<unset>}"
    FAIL=1
  else
    echo "OK      var  $n=$v"
  fi
}
need_file() {
  local n=$1 v=${!1:-}
  need_var "$n"
  if [ -n "$v" ] && [ ! -f "$v" ]; then echo "MISSING file $n=$v"; FAIL=1; fi
}
need_dir() {
  local n=$1 v=${!1:-}
  need_var "$n"
  if [ -n "$v" ] && [ ! -d "$v" ]; then echo "MISSING dir  $n=$v"; FAIL=1; fi
}
need_cards() {
  need_var CRITIC_CARD_ROOTS
  local oldifs=$IFS c
  IFS=:
  for c in ${CRITIC_CARD_ROOTS:-}; do
    if [ ! -d "$c" ]; then echo "MISSING dir  critic cards: $c"; FAIL=1; else echo "OK      dir  critic cards: $c"; fi
  done
  IFS=$oldifs
}
common_base() {
  need_var GROUNDING_ACCOUNT
  need_dir GROUNDING_REPO
  need_dir GROUNDING_DATA_ROOT
  need_dir AGENT_CKPT_ROOT
  need_dir GROUNDING_CONDA_ROOT
}
common_train() {
  common_base
  need_var GROUNDING_TRAIN_PARTITIONS
  need_file GROUNDING_CONTAINER_IMAGE
}
common_eval() {
  common_base
  need_var GROUNDING_EVAL_PARTITIONS
  need_dir VGGT_OMEGA_ROOT
  need_file VGGT_OMEGA_CKPT
  need_file S4VY_CKPT
  need_file S4VY_PROMPT_CKPT
  need_dir SEGVGGT_ROOT
}

case "$TARGET" in
  grounder-4b)
    common_train; need_dir QWEN3_VL_4B; need_file GROUNDER_BIG_MANIFEST ;;
  critic-2b)
    common_train; need_dir QWEN3_VL_2B; need_cards ;;
  eval-spp)
    common_eval; need_dir QWEN3_VL_2B; need_file SPP_EVAL_MANIFEST; need_dir INSSCENE_SPP_ROOT; need_dir SPACEFORMER_SPP_ROOT ;;
  eval-video)
    common_eval; need_dir QWEN3_VL_2B
    if [ ! -d "${GROUNDING_DATA_ROOT:-}/RefVOS" ]; then echo "MISSING dir  ${GROUNDING_DATA_ROOT:-}/RefVOS"; FAIL=1; fi ;;
  *)
    echo "Usage: $0 {grounder-4b|critic-2b|eval-spp|eval-video}" >&2
    exit 2 ;;
esac

if [ "$FAIL" -ne 0 ]; then
  echo "PREFLIGHT FAILED: $TARGET"
  exit 1
fi
echo "PREFLIGHT OK: $TARGET"
