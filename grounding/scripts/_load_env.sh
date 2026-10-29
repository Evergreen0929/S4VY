#!/bin/bash
# Shared by all grounding launchers. It may be sourced more than once.
_GROUNDING_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GROUNDING_REPO="${GROUNDING_REPO:-$(cd "${_GROUNDING_SCRIPT_DIR}/.." && pwd)}"
export SEG4D_REPO="${SEG4D_REPO:-$(cd "${GROUNDING_REPO}/.." && pwd)}"

_GROUNDING_ENV_FILE="${GROUNDING_ENV:-${GROUNDING_REPO}/env.sh}"
if [ -f "${_GROUNDING_ENV_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${_GROUNDING_ENV_FILE}"
elif [ -n "${GROUNDING_ENV:-}" ]; then
  echo "ERROR: GROUNDING_ENV does not exist: ${GROUNDING_ENV}" >&2
  return 2 2>/dev/null || exit 2
fi

export GROUNDING_REPO SEG4D_REPO
export GROUNDING_DATA_LOG_ROOT="${GROUNDING_DATA_LOG_ROOT:-${GROUNDING_DATA_ROOT:-/PATH/TO/data}/../data_log}"
export GROUNDING_LOG_ROOT="${GROUNDING_LOG_ROOT:-${GROUNDING_REPO}/logs}"
export GROUNDING_CONDA_ENV="${GROUNDING_CONDA_ENV:-groundsft}"
export GROUNDING_CONTAINER_MOUNTS="${GROUNDING_CONTAINER_MOUNTS:-/data:/data}"
