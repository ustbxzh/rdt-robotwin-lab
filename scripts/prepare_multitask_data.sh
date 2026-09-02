#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:-0}"
SOURCE_ROOT="${2:-${PROJECT_ROOT}/robotwin/data/train_clean}"

bash "${PROJECT_ROOT}/policy/rdt_1b/process_data.sh" \
  RoboTwin \
  rdt_robotwin_multitask \
  arx_x5 \
  joint \
  50 \
  "${SOURCE_ROOT}" \
  --gpu "${GPU_ID}"
