#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_DIR="${PROJECT_ROOT}/experiments/rdt_robotwin_multitask"
GPU_ID="${1:-0}"
SEED="${2:-0}"
STAT_FILE="${EXPERIMENT_DIR}/stats/dataset_stat.json"

# shellcheck disable=SC1091
source "${EXPERIMENT_DIR}/train.env"

if [[ ! -f "${STAT_FILE}" ]]; then
  echo "[train] missing mixed-dataset statistics: ${STAT_FILE}" >&2
  echo "[train] run: bash scripts/compute_dataset_stats.sh" >&2
  exit 1
fi

export RDT_DATASET_STAT_PATH="${STAT_FILE}"

exec bash "${PROJECT_ROOT}/policy/rdt_1b/train.sh" \
  RoboTwin \
  rdt_robotwin_multitask \
  arx_x5 \
  joint \
  "${SEED}" \
  "${GPU_ID}"
