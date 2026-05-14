#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MMYOLO_ROOT="${MMYOLO_ROOT:-/home/cv09f26/group9/mmyolo/mmyolo}"
CONFIG="${CONFIG:-${ROOT_DIR}/configs/sodad-benchmarks/yolov8_n_sodaa.py}"
WORK_DIR="${WORK_DIR:-${ROOT_DIR}/work_dirs/yolov8_n_sodaa_75}"
GPUS="${GPUS:-1}"
CUDA_DEVICES="${CUDA_DEVICES:-}"
ENV_NAME="${ENV_NAME:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
USE_AMP="${USE_AMP:-1}"

# Optional overrides:
#   MMYOLO_ROOT=/home/cv09f26/group9/mmyolo/mmyolo
#   CONFIG=/home/cv09f26/group9/CFINet_vlv/configs/sodad-benchmarks/yolov8_n_sodaa.py
#   WORK_DIR=/home/cv09f26/group9/CFINet_vlv/work_dirs/yolov8_n_sodaa
#   GPUS=2
#   CUDA_DEVICES=0,1
#   ENV_NAME=yolov8eval
#   EXTRA_ARGS="--amp"

if [[ -n "${ENV_NAME}" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.bashrc"
  conda activate "${ENV_NAME}"
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}"
  exit 1
fi

if [[ ! -d "${MMYOLO_ROOT}" ]]; then
  echo "MMYOLO_ROOT not found: ${MMYOLO_ROOT}"
  exit 1
fi

if [[ -n "${CUDA_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
fi

# Mitigate intermittent CUDA OOM after many epochs: reserved memory can fragment so a
# large contiguous alloc (e.g. one_hot in the loss) fails despite "free" VRAM showing.
# Override if needed: PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 ./tools/train_sodaa_yolov8.sh
#export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# Force local MMYOLO package (includes custom SODAADataset adapter).
export PYTHONPATH="${MMYOLO_ROOT}:${PYTHONPATH:-}"

echo "Training YOLOv8-N on SODA-A"
echo "Config: ${CONFIG}"
echo "Work dir: ${WORK_DIR}"
echo "MMYOLO root: ${MMYOLO_ROOT}"
echo "GPUs: ${GPUS}"
if [[ -n "${CUDA_DEVICES}" ]]; then
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
fi
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"

# Build COCO-style merged json files for evaluator if missing.
python3 - <<'PY'
import json
from pathlib import Path

data_root = Path('/home/cv09f26/group9/CFINet/data/SODA-A')
split_root = data_root / 'divData'

def merge_ann_dir(ann_dir: Path, out_json: Path):
    images = []
    annotations = []
    categories = None
    ann_id = 1
    img_id = 1

    for jf in sorted(ann_dir.glob('*.json')):
        d = json.loads(jf.read_text())
        if categories is None:
            categories = d.get('categories', [])

        if 'images' in d:
            im = d['images'][0] if isinstance(d['images'], list) else d['images']
            file_name = im.get('file_name', im.get('filename'))
            width, height = im['width'], im['height']
            cur_img_id = int(im.get('id', img_id))
            img_id = max(img_id, cur_img_id + 1)
        else:
            file_name = d['filename']
            width, height = d['width'], d['height']
            cur_img_id = img_id
            img_id += 1

        images.append(
            dict(id=cur_img_id, file_name=file_name, width=width, height=height))

        for ann in d.get('annotations', []):
            ann = dict(ann)
            if 'category_id' not in ann and 'cat_id' in ann:
                ann['category_id'] = ann['cat_id']
            if 'bbox' not in ann and 'poly' in ann:
                pts = ann['poly']
                xs, ys = pts[0::2], pts[1::2]
                x1, y1 = min(xs), min(ys)
                x2, y2 = max(xs), max(ys)
                ann['bbox'] = [x1, y1, x2 - x1, y2 - y1]
            if 'area' not in ann or not ann['area']:
                x, y, w, h = ann['bbox']
                ann['area'] = w * h
            ann['id'] = ann_id
            ann_id += 1
            ann['image_id'] = cur_img_id
            ann.setdefault('iscrowd', 0)
            ann.pop('cat_id', None)
            annotations.append(ann)

    out = dict(images=images, annotations=annotations, categories=categories or [])
    out_json.write_text(json.dumps(out))
    print(f'Wrote {out_json} ({len(images)} images, {len(annotations)} anns)')

for split in ('val', 'test'):
    ann_dir = split_root / split / 'Annotations'
    out_json = data_root / f'{split}_coco.json'
    if out_json.exists():
        print(f'Using existing {out_json}')
    else:
        merge_ann_dir(ann_dir, out_json)
PY

cd "${MMYOLO_ROOT}"
if [[ "${GPUS}" -gt 1 ]]; then
  bash tools/dist_train.sh "${CONFIG}" "${GPUS}" \
    --work-dir "${WORK_DIR}" ${EXTRA_ARGS}
else
  python tools/train.py "${CONFIG}" \
    --work-dir "${WORK_DIR}" ${EXTRA_ARGS}
fi
