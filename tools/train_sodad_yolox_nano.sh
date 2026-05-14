#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT_DIR}/configs/sodad-benchmarks/yolox_nano_70e.py"
WORK_DIR="${ROOT_DIR}/work_dirs/yolox_nano_sodad_70e"

# Optional overrides:
#   DATA_ROOT=/path/to/CFINet/data
#   GPUS=1 (preferred)
#   GPU=1  (alias)
#   SEED=0
#   EXTRA_ARGS="--resume-from /path/to/latest.pth"
DATA_ROOT="${DATA_ROOT:-/home/cv09f26/group9/CFINet/data}"
GPUS="${GPUS:-${GPU:-1}}"
SEED="${SEED:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128


if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}"
  exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT not found: ${DATA_ROOT}"
  exit 1
fi

echo "Training YOLOX-Nano on SODA-D"
echo "Config: ${CONFIG}"
echo "Work dir: ${WORK_DIR}"
echo "Data root: ${DATA_ROOT}"
echo "GPUs: ${GPUS}"

if [[ "${GPUS}" -gt 1 ]]; then
  bash "${ROOT_DIR}/tools/dist_train.sh" "${CONFIG}" "${GPUS}" \
    --work-dir "${WORK_DIR}" \
    --cfg-options data_root="${DATA_ROOT}/" \
    --seed "${SEED}" ${EXTRA_ARGS}
else
  python "${ROOT_DIR}/tools/train.py" "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --gpu-id 0 \
    --cfg-options data_root="${DATA_ROOT}/" \
    --seed "${SEED}" ${EXTRA_ARGS}
fi

