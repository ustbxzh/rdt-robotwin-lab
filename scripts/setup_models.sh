#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="${PROJECT_ROOT}/policy/rdt_1b/weights/RDT"
MODE="${1:-check}"

RDT_REPO="robotics-diffusion-transformer/rdt-1b"
T5_REPO="google/t5-v1_1-xxl"
SIGLIP_REPO="google/siglip-so400m-patch14-384"

mkdir -p "${WEIGHTS_DIR}"

if [[ "${MODE}" == "download" ]]; then
  if ! command -v hf >/dev/null 2>&1; then
    echo "[models] Hugging Face CLI 'hf' not found. Install huggingface_hub first." >&2
    exit 1
  fi
  hf download "${RDT_REPO}" --local-dir "${WEIGHTS_DIR}/rdt-1b"
  hf download "${T5_REPO}" --local-dir "${WEIGHTS_DIR}/t5-v1_1-xxl"
  hf download "${SIGLIP_REPO}" --local-dir "${WEIGHTS_DIR}/siglip-so400m-patch14-384"
fi

missing=0
for item in rdt-1b t5-v1_1-xxl siglip-so400m-patch14-384; do
  if [[ -e "${WEIGHTS_DIR}/${item}" ]]; then
    echo "[models] OK      ${WEIGHTS_DIR}/${item}"
  else
    echo "[models] MISSING ${WEIGHTS_DIR}/${item}"
    missing=1
  fi
done

cat <<'EOF'

Official model sources:
  RDT-1B : https://huggingface.co/robotics-diffusion-transformer/rdt-1b
  T5-XXL : https://huggingface.co/google/t5-v1_1-xxl
  SigLIP : https://huggingface.co/google/siglip-so400m-patch14-384

Use `bash scripts/setup_models.sh download` for an explicit Hugging Face download,
or symlink already-downloaded model directories into policy/rdt_1b/weights/RDT/.
EOF

exit "${missing}"
