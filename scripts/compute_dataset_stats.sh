#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_TAG="RoboTwin-rdt_robotwin_multitask-arx_x5-joint"
DATA_DIR="${PROJECT_ROOT}/policy/rdt_1b/data/${DATA_TAG}"
OUT_DIR="${PROJECT_ROOT}/experiments/rdt_robotwin_multitask/stats"
OUT_FILE="${OUT_DIR}/dataset_stat.json"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "[stats] missing prepared dataset: ${DATA_DIR}" >&2
  echo "[stats] run: bash scripts/prepare_multitask_data.sh" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
export RDT_HDF5_DIR="${DATA_DIR}"
export RDT_DATASET_NAME="robotwin_arx_x5_multitask"

cd "${PROJECT_ROOT}/rdt"
PYTHONPATH="${PROJECT_ROOT}/rdt:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
python -m data.compute_dataset_stat_hdf5 --save_path "${OUT_FILE}"

echo "[stats] wrote ${OUT_FILE}"
