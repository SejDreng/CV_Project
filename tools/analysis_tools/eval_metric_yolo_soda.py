# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import importlib.util
import math
import time
from pathlib import Path

import numpy as np
from mmengine.config import Config, DictAction
from mmengine.fileio import load as mmengine_load
from pycocotools.coco import COCO


SODA_METRIC_NAMES = {
    'AP': 0,
    'AP_50': 1,
    'AP_75': 2,
    'AP_eS': 3,
    'AP_rS': 4,
    'AP_gS': 5,
    'AP_Normal': 6,
    'AR@100': 7,
    'AR@300': 8,
    'AR@1000': 9,
    'AR_eS@1000': 10,
    'AR_rS@1000': 11,
    'AR_gS@1000': 12,
    'AR_Normal@1000': 13,
}


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO pkl -> SODA metrics')
    parser.add_argument('config', help='model config path')
    parser.add_argument('pkl_results', help='prediction pkl path')
    parser.add_argument('--eval', type=str, nargs='+', default=['bbox'])
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--chunk-size', type=int, default=200)
    parser.add_argument('--progress-interval', type=int, default=20)
    parser.add_argument('--score-thr', type=float, default=0.0)
    parser.add_argument('--max-images', type=int, default=0)
    return parser.parse_args()


def format_seconds(seconds):
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f'{hours}h {minutes:02d}m {secs:02d}s'
    if minutes:
        return f'{minutes}m {secs:02d}s'
    return f'{secs}s'


def load_soda_eval_class():
    base_dir = Path(__file__).resolve().parents[2]
    sodaeval_path = base_dir / 'mmdet' / 'datasets' / 'sodad_eval' / 'sodadeval.py'
    spec = importlib.util.spec_from_file_location('sodadeval_module', str(sodaeval_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Failed to load SODADeval from {sodaeval_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SODADeval


def tensor_to_numpy(x):
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def build_class_names(cfg):
    if cfg.get('metainfo', None) and cfg.metainfo.get('classes', None):
        return tuple(cfg.metainfo['classes'])
    if cfg.get('class_name', None):
        return tuple(cfg.class_name)
    return None


def build_label_to_cat_id(coco_gt, class_names):
    categories = coco_gt.loadCats(coco_gt.getCatIds())
    name_to_id = {cat['name']: cat['id'] for cat in categories}
    sorted_cat_ids = sorted(coco_gt.getCatIds())
    if class_names is None:
        return sorted_cat_ids

    mapped = []
    for cls_name in class_names:
        if cls_name in name_to_id:
            mapped.append(name_to_id[cls_name])
    if len(mapped) != len(class_names):
        if len(sorted_cat_ids) != len(class_names):
            raise ValueError('Class name/category mismatch for SODA evaluation.')
        print('Warning: class-name mapping incomplete, fallback to sorted ids.')
        return sorted_cat_ids
    return mapped


def convert_outputs(outputs, label_to_cat_id, chunk_size, progress_interval, score_thr):
    detections = []
    unique_img_ids = set()
    total = len(outputs)
    num_chunks = math.ceil(total / chunk_size)
    start = time.perf_counter()

    for chunk_idx in range(num_chunks):
        s = chunk_idx * chunk_size
        e = min((chunk_idx + 1) * chunk_size, total)
        for sample in outputs[s:e]:
            img_id = int(sample['img_id'])
            unique_img_ids.add(img_id)
            pred = sample.get('pred_instances', {})
            bboxes = tensor_to_numpy(pred.get('bboxes', np.empty((0, 4), dtype=np.float32)))
            scores = tensor_to_numpy(pred.get('scores', np.empty((0,), dtype=np.float32)))
            labels = tensor_to_numpy(pred.get('labels', np.empty((0,), dtype=np.int64)))
            for bbox, score, label in zip(bboxes, scores, labels):
                if float(score) < score_thr:
                    continue
                label_idx = int(label)
                if label_idx < 0 or label_idx >= len(label_to_cat_id):
                    continue
                x1, y1, x2, y2 = bbox.tolist()
                detections.append({
                    'image_id': img_id,
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'score': float(score),
                    'category_id': int(label_to_cat_id[label_idx]),
                })

        done = chunk_idx + 1
        if done == 1 or done == num_chunks or done % max(1, progress_interval) == 0:
            elapsed = time.perf_counter() - start
            eta = (elapsed / done) * (num_chunks - done)
            print(f'  - converted chunk {done}/{num_chunks} ({e}/{total}) | '
                  f'elapsed: {format_seconds(elapsed)} | ETA: {format_seconds(eta)}')
    return detections, sorted(unique_img_ids)


def main():
    args = parse_args()
    print(f'Loading config: {args.config}')
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    ann_file = cfg.get('test_evaluator', {}).get('ann_file')
    if not ann_file:
        raise ValueError('test_evaluator.ann_file is required.')

    print('[1/5] Loading SODA evaluator...')
    SODADeval = load_soda_eval_class()

    print('[2/5] Loading predictions...')
    outputs = mmengine_load(args.pkl_results)
    print(f'Loaded {len(outputs)} predictions from pkl.')

    print(f'[3/5] Loading GT annotations: {ann_file}')
    coco_gt = COCO(ann_file)
    label_to_cat_id = build_label_to_cat_id(coco_gt, build_class_names(cfg))
    print(f'Using {len(label_to_cat_id)} category ids for label mapping.')

    print('[4/5] Converting model outputs to COCO detections...')
    print(f'Applying score threshold: {args.score_thr}')
    detections, img_ids = convert_outputs(
        outputs=outputs,
        label_to_cat_id=label_to_cat_id,
        chunk_size=max(1, args.chunk_size),
        progress_interval=max(1, args.progress_interval),
        score_thr=args.score_thr)
    print(f'Converted {len(detections)} detections over {len(img_ids)} images.')
    if args.max_images > 0:
        img_ids = img_ids[:args.max_images]
        print(f'Limiting evaluation to first {len(img_ids)} images.')

    print('[5/5] Running SODA evaluation...')
    coco_dt = coco_gt.loadRes(detections) if detections else coco_gt.loadRes([])
    soda_eval = SODADeval(coco_gt, coco_dt, 'bbox')
    soda_eval.params.catIds = label_to_cat_id
    soda_eval.params.imgIds = img_ids if img_ids else sorted(coco_gt.getImgIds())
    soda_eval.params.maxDets = [100, 300, 1000]
    soda_eval.params.iouThrs = np.linspace(.5, 0.95, 10, endpoint=True)
    soda_eval.evaluate()
    soda_eval.accumulate()
    soda_eval.summarize()

    metrics = {k: float(f'{soda_eval.stats[i]:.3f}') for k, i in SODA_METRIC_NAMES.items()}
    metrics['AP_PN'] = metrics['AP_Normal']
    metrics['AR_PN@1000'] = metrics['AR_Normal@1000']
    print('SODA metrics:', metrics)
    print('mAP copy-paste:', (
        f'{metrics["AP"]:.3f} {metrics["AP_50"]:.3f} {metrics["AP_75"]:.3f} '
        f'{metrics["AP_eS"]:.3f} {metrics["AP_rS"]:.3f} {metrics["AP_gS"]:.3f} '
        f'{metrics["AP_Normal"]:.3f}'
    ))


if __name__ == '__main__':
    main()
