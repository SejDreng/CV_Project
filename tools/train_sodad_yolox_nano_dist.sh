#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT_DIR}/configs/sodad-benchmarks/yolox_nano_70e.py"
WORK_DIR="${ROOT_DIR}/work_dirs/yolox_nano_sodad_70e"

# Optional overrides:
#   GPUS=2
#   SEED=0
#   PORT=29500
#   EXTRA_ARGS="--resume-from /path/to/latest.pth"
GPUS="${GPUS:-${GPU:-2}}"
SEED="${SEED:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}"
  exit 1
fi

if [[ "${GPUS}" -lt 2 ]]; then
  echo "This script is for distributed training. Use GPUS>=2 (current: ${GPUS})."
  exit 1
fi

echo "Distributed training YOLOX-Nano on SODA-D"
echo "Config: ${CONFIG}"
echo "Work dir: ${WORK_DIR}"
echo "GPUs: ${GPUS}"
echo "PORT: ${PORT:-29500}"

bash "${ROOT_DIR}/tools/dist_train.sh" "${CONFIG}" "${GPUS}" \
  --work-dir "${WORK_DIR}" \
  --seed "${SEED}" ${EXTRA_ARGS}

