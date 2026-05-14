#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MMYOLO_ROOT="${MMYOLO_ROOT:-/home/cv09f26/group9/mmyolo/mmyolo}"
MMYOLO_PYTHON="${MMYOLO_PYTHON:-python}"
CONFIG="${ROOT_DIR}/configs/sodad-benchmarks/yolov8_n_sodad.py"
WORK_DIR="${ROOT_DIR}/work_dirs/yolov8_n_sodad"

# Optional overrides:
#   GPUS=1 (or GPU=1 alias)
#   SEED=0
#   EXTRA_ARGS="--resume /path/to/checkpoint.pth"
GPUS="${GPUS:-${GPU:-1}}"
SEED="${SEED:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

if [[ ! -d "${MMYOLO_ROOT}" ]]; then
  echo "MMYOLO_ROOT not found: ${MMYOLO_ROOT}"
  exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}"
  exit 1
fi

echo "Training YOLOv8-n on SODA-D via MMYOLO"
echo "MMYOLO root: ${MMYOLO_ROOT}"
echo "Python: ${MMYOLO_PYTHON}"
echo "Config: ${CONFIG}"
echo "Work dir: ${WORK_DIR}"
echo "GPUs: ${GPUS}"

if [[ "${GPUS}" -gt 1 ]]; then
  bash "${MMYOLO_ROOT}/tools/dist_train.sh" "${CONFIG}" "${GPUS}" \
    --work-dir "${WORK_DIR}" \
    --cfg-options randomness.seed="${SEED}" ${EXTRA_ARGS}
else
  "${MMYOLO_PYTHON}" "${MMYOLO_ROOT}/tools/train.py" "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --cfg-options randomness.seed="${SEED}" ${EXTRA_ARGS}
fi

