#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <checkpoint> [clean|randomized|all] [policy_gpu] [env_gpu] [seed]" >&2
  exit 1
fi

CHECKPOINT=$1
DOMAIN="${2:-all}"
POLICY_GPU="${3:-0}"
ENV_GPU="${4:-0}"
SEED="${5:-0}"
POLICY_ENV="${RDT_POLICY_ENV:-rdt_1b}"
ROBOTWIN_ENV="${ROBOTWIN_ENV:-robotwin}"

TASKS=(
  adjust_bottle
  lift_pot
  handover_block
  blocks_ranking_size
)

case "${DOMAIN}" in
  clean) DOMAINS=(clean) ;;
  randomized) DOMAINS=(randomized) ;;
  all) DOMAINS=(clean randomized) ;;
  *) echo "Unknown domain: ${DOMAIN}" >&2; exit 1 ;;
esac

for domain in "${DOMAINS[@]}"; do
  export ROBOTWIN_TASK_CONFIG="eval_${domain}"
  export ROBOTWIN_TEST_NUM="${ROBOTWIN_TEST_NUM:-100}"
  echo "[eval-suite] domain=${domain}, checkpoint=${CHECKPOINT}"

  for task in "${TASKS[@]}"; do
    echo "[eval-suite] task=${task}"
    bash "${PROJECT_ROOT}/scripts/eval_robotwin.sh" \
      RoboTwin \
      "${task}" \
      "${CHECKPOINT}" \
      arx_x5 \
      joint \
      "${SEED}" \
      "${POLICY_GPU}" \
      "${ENV_GPU}" \
      "${POLICY_ENV}" \
      "${ROBOTWIN_ENV}"
  done
done
