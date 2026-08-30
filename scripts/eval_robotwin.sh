#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ $# -ne 10 ]]; then
    echo "Usage: $0 <bench> <task> <checkpoint> <env_cfg> <action_type> <seed> <policy_gpu> <env_gpu> <policy_env> <robotwin_env>" >&2
    exit 1
fi

exec bash "${PROJECT_ROOT}/policy/rdt_1b/eval.sh" "$@"
