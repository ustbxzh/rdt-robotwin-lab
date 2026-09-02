#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:-0}"
TASK_CONFIG="${ROBOTWIN_TRAIN_CONFIG:-train_clean}"

TASKS=(
  adjust_bottle
  lift_pot
  handover_block
  blocks_ranking_size
)

for task in "${TASKS[@]}"; do
  echo "[expert-suite] collecting ${task} with ${TASK_CONFIG} on GPU ${GPU_ID}"
  bash "${PROJECT_ROOT}/scripts/collect_demo.sh" "${task}" "${TASK_CONFIG}" "${GPU_ID}"
done

echo "[expert-suite] complete: ${#TASKS[@]} tasks"
