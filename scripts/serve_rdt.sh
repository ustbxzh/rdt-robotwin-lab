#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec python -m policy.xpolicylab.setup_policy_server \
    --config_path "${PROJECT_ROOT}/policy/rdt_1b/deploy.yml" \
    "$@"
