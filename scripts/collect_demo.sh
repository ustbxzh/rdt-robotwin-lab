#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <task_name> <task_config> <gpu_id>" >&2
    exit 1
fi

if [[ ! -d "${ROBOTWIN_ASSETS_ROOT:-${PROJECT_ROOT}/robotwin/assets}" ]]; then
    echo "RoboTwin assets not found. Set ROBOTWIN_ASSETS_ROOT or create robotwin/assets symlink." >&2
    exit 1
fi

cd "${PROJECT_ROOT}/robotwin"
exec bash collect_data.sh "$@"
