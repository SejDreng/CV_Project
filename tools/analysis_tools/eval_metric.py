# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import importlib.util
import math
import time
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO

try:
    from mmengine.config import Config, DictAction
    from mmengine.evaluator import Evaluator
    from mmengine.fileio import load as mmengine_load
    from mmengine.registry import init_default_scope
    HAS_MMENGINE = True
except ImportError:
    HAS_MMENGINE = False
    import mmcv
    from mmcv import Config, DictAction


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
    'AR_Normal@1000': 13
}


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metrics from pkl results.')
    parser.add_argument('config', help='config of the model')
    parser.add_argument('pkl_results', help='results in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='format output results without evaluation')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, e.g. "bbox", "segm", "proposal"')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override settings in used config, key=value format')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for dataset/evaluator')
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=200,
        help='chunk size for mmengine offline evaluation')
    parser.add_argument(
        '--progress-interval',
        type=int,
        default=20,
        help='print progress every N chunks')
    parser.add_argument(
        '--soda-eval',
        action='store_true',
        help='use SODADeval for MMYOLO/MMEngine outputs to report AP_eS/AP_rS/'
        'AP_gS/AP_Normal metrics')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.0,
        help='filter detections below this score before SODA evaluation')
    parser.add_argument(
        '--max-images',
        type=int,
        default=0,
        help='limit evaluated images for quick smoke testing (0 means all)')
    args = parser.parse_args()
    return args


def format_seconds(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f'{hours:d}h {minutes:02d}m {secs:02d}s'
    if minutes > 0:
        return f'{minutes:d}m {secs:02d}s'
    return f'{secs:d}s'


def is_mmengine_style_cfg(cfg):
    return ('test_dataloader' in cfg or cfg.get('default_scope', None) == 'mmyolo')


def build_class_names(cfg):
    if cfg.get('metainfo', None) and cfg.metainfo.get('classes', None):
        return tuple(cfg.metainfo['classes'])
    if cfg.get('class_name', None):
        return tuple(cfg.class_name)
    return None


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


def build_label_to_cat_id(coco_gt, class_names):
    categories = coco_gt.loadCats(coco_gt.getCatIds())
    name_to_id = {cat['name']: cat['id'] for cat in categories}
    sorted_cat_ids = sorted(coco_gt.getCatIds())

    if class_names is None:
        return sorted_cat_ids

    label_to_cat_id = []
    missing = []
    for cls_name in class_names:
        if cls_name in name_to_id:
            label_to_cat_id.append(name_to_id[cls_name])
        else:
            missing.append(cls_name)

    if missing:
        if len(sorted_cat_ids) != len(class_names):
            raise ValueError(
                'Failed to map class names to COCO category ids.\n'
                f'Missing class names: {missing}\n'
                f'Class count in config: {len(class_names)}, '
                f'category count in ann file: {len(sorted_cat_ids)}')
        print('Warning: class-name mapping incomplete. Falling back to sorted '
              'COCO category ids by label index.')
        return sorted_cat_ids

    return label_to_cat_id


def convert_mmengine_outputs_to_coco_dets(outputs,
                                          label_to_cat_id,
                                          progress_interval,
                                          chunk_size,
                                          score_thr):
    detections = []
    total = len(outputs)
    num_chunks = math.ceil(total / chunk_size)
    start_time = time.perf_counter()
    unique_img_ids = set()

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min((chunk_idx + 1) * chunk_size, total)
        chunk = outputs[start:end]
        for sample in chunk:
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

        done_chunks = chunk_idx + 1
        should_print = (
            done_chunks == 1
            or done_chunks == num_chunks
            or done_chunks % max(1, progress_interval) == 0
        )
        if should_print:
            elapsed = time.perf_counter() - start_time
            avg_chunk = elapsed / done_chunks
            eta = avg_chunk * (num_chunks - done_chunks)
            print(f'  - converted chunk {done_chunks}/{num_chunks} '
                  f'({end}/{total} samples) | elapsed: {format_seconds(elapsed)} '
                  f'| ETA: {format_seconds(eta)}')

    return detections, sorted(unique_img_ids)


def eval_mmengine_with_soda(cfg, args):
    print('[1/5] Loading SODA evaluator...')
    SODADeval = load_soda_eval_class()

    print('[2/5] Loading predictions...')
    outputs = mmengine_load(args.pkl_results)
    total = len(outputs)
    print(f'Loaded {total} predictions from pkl.')
    if total == 0:
        raise ValueError('No predictions found in pkl file.')

    ann_file = cfg.get('test_evaluator', {}).get('ann_file', None)
    if ann_file is None:
        raise ValueError('test_evaluator.ann_file not found in config; '
                         'required for --soda-eval mode.')

    print(f'[3/5] Loading GT annotations: {ann_file}')
    coco_gt = COCO(ann_file)
    class_names = build_class_names(cfg)
    label_to_cat_id = build_label_to_cat_id(coco_gt, class_names)
    print(f'Using {len(label_to_cat_id)} category ids for label mapping.')

    print('[4/5] Converting model outputs to COCO detections...')
    print(f'Applying score threshold: {args.score_thr}')
    detections, img_ids = convert_mmengine_outputs_to_coco_dets(
        outputs=outputs,
        label_to_cat_id=label_to_cat_id,
        progress_interval=args.progress_interval,
        chunk_size=max(1, args.chunk_size),
        score_thr=args.score_thr)
    print(f'Converted {len(detections)} detections over {len(img_ids)} images.')
    if args.max_images and args.max_images > 0:
        img_ids = img_ids[:args.max_images]
        print(f'Limiting evaluation to first {len(img_ids)} images '
              f'(max-images={args.max_images})')

    print('[5/5] Running SODA evaluation...')
    coco_dt = coco_gt.loadRes(detections) if detections else coco_gt.loadRes([])
    soda_eval = SODADeval(coco_gt, coco_dt, 'bbox')
    soda_eval.params.catIds = label_to_cat_id
    soda_eval.params.imgIds = img_ids if img_ids else sorted(coco_gt.getImgIds())
    soda_eval.params.maxDets = [100, 300, 1000]
    soda_eval.params.iouThrs = np.linspace(
        .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)

    soda_eval.evaluate()
    soda_eval.accumulate()
    soda_eval.summarize()

    metrics = {
        name: float(f'{soda_eval.stats[idx]:.3f}')
        for name, idx in SODA_METRIC_NAMES.items()
    }
    metrics['AP_PN'] = metrics['AP_Normal']
    metrics['AR_PN@1000'] = metrics['AR_Normal@1000']

    print('SODA metrics:', metrics)
    print('mAP copy-paste:', (
        f'{metrics["AP"]:.3f} {metrics["AP_50"]:.3f} {metrics["AP_75"]:.3f} '
        f'{metrics["AP_eS"]:.3f} {metrics["AP_rS"]:.3f} {metrics["AP_gS"]:.3f} '
        f'{metrics["AP_Normal"]:.3f}'
    ))


def eval_mmengine_results(cfg, args):
    if not HAS_MMENGINE:
        raise RuntimeError(
            'This config is MMEngine/MMYOLO-style, but mmengine is unavailable '
            'in the current environment.')

    if args.soda_eval:
        eval_mmengine_with_soda(cfg, args)
        return

    print('[1/4] Initializing MMEngine scope...')
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    print(f'[2/4] Loading predictions from: {args.pkl_results}')
    outputs = mmengine_load(args.pkl_results)
    total = len(outputs)
    print(f'Loaded {total} predictions.')

    evaluator_cfg = copy.deepcopy(cfg.test_evaluator)
    kwargs = {} if args.eval_options is None else args.eval_options

    if args.eval:
        evaluator_cfg['metric'] = args.eval if len(args.eval) > 1 else args.eval[0]

    if args.format_only:
        evaluator_cfg['format_only'] = True

    evaluator_cfg.update(kwargs)
    evaluator = Evaluator(evaluator_cfg)

    class_names = build_class_names(cfg)
    if class_names is not None:
        evaluator.dataset_meta = {'classes': class_names}
        print(f'Using {len(class_names)} classes from config metainfo.')
    else:
        print('No class metainfo found in config; evaluator will infer defaults.')

    print('[3/4] Processing predictions for metric accumulation...')
    if total == 0:
        raise ValueError('No predictions found in pkl file.')

    chunk_size = max(1, args.chunk_size)
    progress_interval = max(1, args.progress_interval)
    num_chunks = math.ceil(total / chunk_size)
    process_start = time.perf_counter()

    for idx in range(num_chunks):
        start = idx * chunk_size
        end = min((idx + 1) * chunk_size, total)
        evaluator.process(data_samples=outputs[start:end], data_batch=None)

        chunks_done = idx + 1
        should_print = (
            chunks_done % progress_interval == 0
            or chunks_done == num_chunks
            or chunks_done == 1)
        if should_print:
            elapsed = time.perf_counter() - process_start
            avg_chunk_time = elapsed / chunks_done
            chunks_left = num_chunks - chunks_done
            eta_seconds = avg_chunk_time * chunks_left
            print(
                f'  - chunk {chunks_done}/{num_chunks} '
                f'({end}/{total} samples) | elapsed: {format_seconds(elapsed)} '
                f'| ETA: {format_seconds(eta_seconds)}')

    print('[4/4] Computing final metrics...')
    metrics_start = time.perf_counter()
    metrics = evaluator.evaluate(total)
    metrics_elapsed = time.perf_counter() - metrics_start
    print(f'Final metric computation time: {format_seconds(metrics_elapsed)}')
    print(metrics)


def eval_legacy_results(cfg, args):
    import mmcv
    from mmdet.datasets import build_dataset
    from mmdet.utils import replace_cfg_vals, update_data_root

    print('[1/4] Preparing legacy MMDet config...')
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True

    print('[2/4] Building dataset...')
    dataset = build_dataset(cfg.data.test)

    print(f'[3/4] Loading predictions from: {args.pkl_results}')
    outputs = mmcv.load(args.pkl_results)
    print(f'Loaded {len(outputs)} predictions.')

    kwargs = {} if args.eval_options is None else args.eval_options
    if args.format_only:
        print('[4/4] Formatting output results...')
        dataset.format_results(outputs, **kwargs)
        print('Formatting complete.')
    if args.eval:
        print('[4/4] Running evaluation...')
        eval_kwargs = cfg.get('evaluation', {}).copy()
        for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        eval_start = time.perf_counter()
        print(dataset.evaluate(outputs, **eval_kwargs))
        eval_elapsed = time.perf_counter() - eval_start
        print(f'Evaluation time: {format_seconds(eval_elapsed)}')


def main():
    args = parse_args()

    assert args.eval or args.format_only, (
        'Please specify at least one operation with "--eval" or "--format-only".')
    if args.eval and args.format_only:
        raise ValueError('--eval and --format-only cannot be both specified')

    print(f'Loading config: {args.config}')
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    mode = 'MMEngine/MMYOLO' if is_mmengine_style_cfg(cfg) else 'Legacy MMDet'
    print(f'Detected config mode: {mode}')

    overall_start = time.perf_counter()
    if is_mmengine_style_cfg(cfg):
        eval_mmengine_results(cfg, args)
    else:
        eval_legacy_results(cfg, args)
    overall_elapsed = time.perf_counter() - overall_start
    print(f'Total elapsed time: {format_seconds(overall_elapsed)}')


if __name__ == '__main__':
    main()
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import importlib.util
import math
import time
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO

try:
    from mmengine.config import Config, DictAction
    from mmengine.evaluator import Evaluator
    from mmengine.fileio import load as mmengine_load
    from mmengine.registry import init_default_scope
    HAS_MMENGINE = True
except ImportError:
    HAS_MMENGINE = False
    import mmcv
    from mmcv import Config, DictAction


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
    'AR_Normal@1000': 13
}


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metrics from pkl results.')
    parser.add_argument('config', help='config of the model')
    parser.add_argument('pkl_results', help='results in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='format output results without evaluation')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, e.g. "bbox", "segm", "proposal"')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override settings in used config, key=value format')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for dataset/evaluator')
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=200,
        help='chunk size for mmengine offline evaluation')
    parser.add_argument(
        '--progress-interval',
        type=int,
        default=20,
        help='print progress every N chunks')
    parser.add_argument(
        '--soda-eval',
        action='store_true',
        help='use SODADeval for MMYOLO/MMEngine outputs to report AP_eS/AP_rS/'
        'AP_gS/AP_Normal metrics')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.0,
        help='filter detections below this score before SODA evaluation')
    parser.add_argument(
        '--max-images',
        type=int,
        default=0,
        help='limit evaluated images for quick smoke testing (0 means all)')
    args = parser.parse_args()
    return args


def format_seconds(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f'{hours:d}h {minutes:02d}m {secs:02d}s'
    if minutes > 0:
        return f'{minutes:d}m {secs:02d}s'
    return f'{secs:d}s'


def is_mmengine_style_cfg(cfg):
    return ('test_dataloader' in cfg or cfg.get('default_scope', None) == 'mmyolo')


def build_class_names(cfg):
    if cfg.get('metainfo', None) and cfg.metainfo.get('classes', None):
        return tuple(cfg.metainfo['classes'])
    if cfg.get('class_name', None):
        return tuple(cfg.class_name)
    return None


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


def build_label_to_cat_id(coco_gt, class_names):
    categories = coco_gt.loadCats(coco_gt.getCatIds())
    name_to_id = {cat['name']: cat['id'] for cat in categories}
    sorted_cat_ids = sorted(coco_gt.getCatIds())

    if class_names is None:
        return sorted_cat_ids

    label_to_cat_id = []
    missing = []
    for cls_name in class_names:
        if cls_name in name_to_id:
            label_to_cat_id.append(name_to_id[cls_name])
        else:
            missing.append(cls_name)

    if missing:
        if len(sorted_cat_ids) != len(class_names):
            raise ValueError(
                'Failed to map class names to COCO category ids.\n'
                f'Missing class names: {missing}\n'
                f'Class count in config: {len(class_names)}, '
                f'category count in ann file: {len(sorted_cat_ids)}')
        print('Warning: class-name mapping incomplete. Falling back to sorted '
              'COCO category ids by label index.')
        return sorted_cat_ids

    return label_to_cat_id


def convert_mmengine_outputs_to_coco_dets(outputs,
                                          label_to_cat_id,
                                          progress_interval,
                                          chunk_size,
                                          score_thr):
    detections = []
    total = len(outputs)
    num_chunks = math.ceil(total / chunk_size)
    start_time = time.perf_counter()
    unique_img_ids = set()

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min((chunk_idx + 1) * chunk_size, total)
        chunk = outputs[start:end]
        for sample in chunk:
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

        done_chunks = chunk_idx + 1
        should_print = (
            done_chunks == 1
            or done_chunks == num_chunks
            or done_chunks % max(1, progress_interval) == 0
        )
        if should_print:
            elapsed = time.perf_counter() - start_time
            avg_chunk = elapsed / done_chunks
            eta = avg_chunk * (num_chunks - done_chunks)
            print(f'  - converted chunk {done_chunks}/{num_chunks} '
                  f'({end}/{total} samples) | elapsed: {format_seconds(elapsed)} '
                  f'| ETA: {format_seconds(eta)}')

    return detections, sorted(unique_img_ids)


def eval_mmengine_with_soda(cfg, args):
    print('[1/5] Loading SODA evaluator...')
    SODADeval = load_soda_eval_class()

    print('[2/5] Loading predictions...')
    outputs = mmengine_load(args.pkl_results)
    total = len(outputs)
    print(f'Loaded {total} predictions from pkl.')
    if total == 0:
        raise ValueError('No predictions found in pkl file.')

    ann_file = cfg.get('test_evaluator', {}).get('ann_file', None)
    if ann_file is None:
        raise ValueError('test_evaluator.ann_file not found in config; '
                         'required for --soda-eval mode.')

    print(f'[3/5] Loading GT annotations: {ann_file}')
    coco_gt = COCO(ann_file)
    class_names = build_class_names(cfg)
    label_to_cat_id = build_label_to_cat_id(coco_gt, class_names)
    print(f'Using {len(label_to_cat_id)} category ids for label mapping.')

    print('[4/5] Converting model outputs to COCO detections...')
    print(f'Applying score threshold: {args.score_thr}')
    detections, img_ids = convert_mmengine_outputs_to_coco_dets(
        outputs=outputs,
        label_to_cat_id=label_to_cat_id,
        progress_interval=args.progress_interval,
        chunk_size=max(1, args.chunk_size),
        score_thr=args.score_thr)
    print(f'Converted {len(detections)} detections over {len(img_ids)} images.')
    if args.max_images and args.max_images > 0:
        img_ids = img_ids[:args.max_images]
        print(f'Limiting evaluation to first {len(img_ids)} images '
              f'(max-images={args.max_images})')

    print('[5/5] Running SODA evaluation...')
    coco_dt = coco_gt.loadRes(detections) if detections else coco_gt.loadRes([])
    soda_eval = SODADeval(coco_gt, coco_dt, 'bbox')
    soda_eval.params.catIds = label_to_cat_id
    soda_eval.params.imgIds = img_ids if img_ids else sorted(coco_gt.getImgIds())
    soda_eval.params.maxDets = [100, 300, 1000]
    soda_eval.params.iouThrs = np.linspace(
        .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)

    soda_eval.evaluate()
    soda_eval.accumulate()
    soda_eval.summarize()

    metrics = {
        name: float(f'{soda_eval.stats[idx]:.3f}')
        for name, idx in SODA_METRIC_NAMES.items()
    }
    # Alias to match your table naming.
    metrics['AP_PN'] = metrics['AP_Normal']
    metrics['AR_PN@1000'] = metrics['AR_Normal@1000']

    print('SODA metrics:', metrics)
    print('mAP copy-paste:', (
        f'{metrics["AP"]:.3f} {metrics["AP_50"]:.3f} {metrics["AP_75"]:.3f} '
        f'{metrics["AP_eS"]:.3f} {metrics["AP_rS"]:.3f} {metrics["AP_gS"]:.3f} '
        f'{metrics["AP_Normal"]:.3f}'
    ))


def eval_mmengine_results(cfg, args):
    if not HAS_MMENGINE:
        raise RuntimeError(
            'This config is MMEngine/MMYOLO-style, but mmengine is unavailable '
            'in the current environment.')

    if args.soda_eval:
        eval_mmengine_with_soda(cfg, args)
        return

    print('[1/4] Initializing MMEngine scope...')
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    print(f'[2/4] Loading predictions from: {args.pkl_results}')
    outputs = mmengine_load(args.pkl_results)
    total = len(outputs)
    print(f'Loaded {total} predictions.')

    evaluator_cfg = copy.deepcopy(cfg.test_evaluator)
    kwargs = {} if args.eval_options is None else args.eval_options

    if args.eval:
        evaluator_cfg['metric'] = args.eval if len(args.eval) > 1 else args.eval[0]

    if args.format_only:
        evaluator_cfg['format_only'] = True

    evaluator_cfg.update(kwargs)
    evaluator = Evaluator(evaluator_cfg)

    class_names = build_class_names(cfg)
    if class_names is not None:
        evaluator.dataset_meta = {'classes': class_names}
        print(f'Using {len(class_names)} classes from config metainfo.')
    else:
        print('No class metainfo found in config; evaluator will infer defaults.')

    print('[3/4] Processing predictions for metric accumulation...')
    if total == 0:
        raise ValueError('No predictions found in pkl file.')

    chunk_size = max(1, args.chunk_size)
    progress_interval = max(1, args.progress_interval)
    num_chunks = math.ceil(total / chunk_size)
    process_start = time.perf_counter()

    for idx in range(num_chunks):
        start = idx * chunk_size
        end = min((idx + 1) * chunk_size, total)
        evaluator.process(data_samples=outputs[start:end], data_batch=None)

        chunks_done = idx + 1
        should_print = (
            chunks_done % progress_interval == 0
            or chunks_done == num_chunks
            or chunks_done == 1)
        if should_print:
            elapsed = time.perf_counter() - process_start
            avg_chunk_time = elapsed / chunks_done
            chunks_left = num_chunks - chunks_done
            eta_seconds = avg_chunk_time * chunks_left
            print(
                f'  - chunk {chunks_done}/{num_chunks} '
                f'({end}/{total} samples) | elapsed: {format_seconds(elapsed)} '
                f'| ETA: {format_seconds(eta_seconds)}')

    print('[4/4] Computing final metrics...')
    metrics_start = time.perf_counter()
    metrics = evaluator.evaluate(total)
    metrics_elapsed = time.perf_counter() - metrics_start
    print(f'Final metric computation time: {format_seconds(metrics_elapsed)}')
    print(metrics)


def eval_legacy_results(cfg, args):
    import mmcv
    from mmdet.datasets import build_dataset
    from mmdet.utils import replace_cfg_vals, update_data_root

    print('[1/4] Preparing legacy MMDet config...')
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True

    print('[2/4] Building dataset...')
    dataset = build_dataset(cfg.data.test)

    print(f'[3/4] Loading predictions from: {args.pkl_results}')
    outputs = mmcv.load(args.pkl_results)
    print(f'Loaded {len(outputs)} predictions.')

    kwargs = {} if args.eval_options is None else args.eval_options
    if args.format_only:
        print('[4/4] Formatting output results...')
        dataset.format_results(outputs, **kwargs)
        print('Formatting complete.')
    if args.eval:
        print('[4/4] Running evaluation...')
        eval_kwargs = cfg.get('evaluation', {}).copy()
        for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        eval_start = time.perf_counter()
        print(dataset.evaluate(outputs, **eval_kwargs))
        eval_elapsed = time.perf_counter() - eval_start
        print(f'Evaluation time: {format_seconds(eval_elapsed)}')


def main():
    args = parse_args()

    assert args.eval or args.format_only, (
        'Please specify at least one operation with "--eval" or "--format-only".')
    if args.eval and args.format_only:
        raise ValueError('--eval and --format-only cannot be both specified')

    print(f'Loading config: {args.config}')
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    mode = 'MMEngine/MMYOLO' if is_mmengine_style_cfg(cfg) else 'Legacy MMDet'
    print(f'Detected config mode: {mode}')

    overall_start = time.perf_counter()
    if is_mmengine_style_cfg(cfg):
        eval_mmengine_results(cfg, args)
    else:
        eval_legacy_results(cfg, args)
    overall_elapsed = time.perf_counter() - overall_start
    print(f'Total elapsed time: {format_seconds(overall_elapsed)}')


if __name__ == '__main__':
    main()
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import math
import time

try:
    from mmengine.config import Config, DictAction
    from mmengine.evaluator import Evaluator
    from mmengine.fileio import load as mmengine_load
    from mmengine.registry import init_default_scope
    HAS_MMENGINE = True
except ImportError:
    HAS_MMENGINE = False
    import mmcv
    from mmcv import Config, DictAction


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metrics from pkl results.')
    parser.add_argument('config', help='config of the model')
    parser.add_argument('pkl_results', help='results in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='format output results without evaluation')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, e.g. "bbox", "segm", "proposal"')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override settings in used config, key=value format')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for dataset/evaluator')
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=200,
        help='chunk size for mmengine offline evaluation')
    parser.add_argument(
        '--progress-interval',
        type=int,
        default=1,
        help='print progress every N chunks')
    args = parser.parse_args()
    return args


def format_seconds(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f'{hours:d}h {minutes:02d}m {secs:02d}s'
    if minutes > 0:
        return f'{minutes:d}m {secs:02d}s'
    return f'{secs:d}s'


def is_mmengine_style_cfg(cfg):
    return ('test_dataloader' in cfg or cfg.get('default_scope', None) == 'mmyolo')


def build_dataset_meta(cfg):
    classes = None
    if cfg.get('metainfo', None) and cfg.metainfo.get('classes', None):
        classes = tuple(cfg.metainfo['classes'])
    elif cfg.get('class_name', None):
        classes = tuple(cfg.class_name)

    if classes is None:
        return None
    return {'classes': classes}


def eval_mmengine_results(cfg, args):
    if not HAS_MMENGINE:
        raise RuntimeError(
            'This config is MMEngine/MMYOLO-style, but mmengine is unavailable '
            'in the current environment.')

    print('[1/4] Initializing MMEngine scope...')
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    print(f'[2/4] Loading predictions from: {args.pkl_results}')
    outputs = mmengine_load(args.pkl_results)
    total = len(outputs)
    print(f'Loaded {total} predictions.')

    evaluator_cfg = copy.deepcopy(cfg.test_evaluator)
    kwargs = {} if args.eval_options is None else args.eval_options

    if args.eval:
        # Keep legacy behavior: --eval overrides config metric.
        evaluator_cfg['metric'] = args.eval if len(args.eval) > 1 else args.eval[0]

    if args.format_only:
        evaluator_cfg['format_only'] = True

    evaluator_cfg.update(kwargs)
    evaluator = Evaluator(evaluator_cfg)

    dataset_meta = build_dataset_meta(cfg)
    if dataset_meta is not None:
        evaluator.dataset_meta = dataset_meta
        print(f'Using {len(dataset_meta["classes"])} classes from config metainfo.')
    else:
        print('No class metainfo found in config; evaluator will infer defaults.')

    print('[3/4] Processing predictions for metric accumulation...')
    if total == 0:
        raise ValueError('No predictions found in pkl file.')

    chunk_size = max(1, args.chunk_size)
    progress_interval = max(1, args.progress_interval)
    num_chunks = math.ceil(total / chunk_size)
    process_start = time.perf_counter()

    for idx in range(num_chunks):
        start = idx * chunk_size
        end = min((idx + 1) * chunk_size, total)
        evaluator.process(data_samples=outputs[start:end], data_batch=None)

        chunks_done = idx + 1
        should_print = (
            chunks_done % progress_interval == 0
            or chunks_done == num_chunks
            or chunks_done == 1)
        if should_print:
            elapsed = time.perf_counter() - process_start
            avg_chunk_time = elapsed / chunks_done
            chunks_left = num_chunks - chunks_done
            eta_seconds = avg_chunk_time * chunks_left
            print(
                f'  - chunk {chunks_done}/{num_chunks} '
                f'({end}/{total} samples) | elapsed: {format_seconds(elapsed)} '
                f'| ETA: {format_seconds(eta_seconds)}')

    print('[4/4] Computing final metrics...')
    metrics_start = time.perf_counter()
    metrics = evaluator.evaluate(total)
    metrics_elapsed = time.perf_counter() - metrics_start
    print(f'Final metric computation time: {format_seconds(metrics_elapsed)}')
    print(metrics)


def eval_legacy_results(cfg, args):
    import mmcv
    from mmdet.datasets import build_dataset
    from mmdet.utils import replace_cfg_vals, update_data_root

    print('[1/4] Preparing legacy MMDet config...')
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True

    print('[2/4] Building dataset...')
    dataset = build_dataset(cfg.data.test)

    print(f'[3/4] Loading predictions from: {args.pkl_results}')
    outputs = mmcv.load(args.pkl_results)
    print(f'Loaded {len(outputs)} predictions.')

    kwargs = {} if args.eval_options is None else args.eval_options
    if args.format_only:
        print('[4/4] Formatting output results...')
        dataset.format_results(outputs, **kwargs)
        print('Formatting complete.')
    if args.eval:
        print('[4/4] Running evaluation...')
        eval_kwargs = cfg.get('evaluation', {}).copy()
        # Hard-code way to remove EvalHook args.
        for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule'
        ]:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        eval_start = time.perf_counter()
        print(dataset.evaluate(outputs, **eval_kwargs))
        eval_elapsed = time.perf_counter() - eval_start
        print(f'Evaluation time: {format_seconds(eval_elapsed)}')


def main():
    args = parse_args()

    assert args.eval or args.format_only, (
        'Please specify at least one operation with "--eval" or "--format-only".')
    if args.eval and args.format_only:
        raise ValueError('--eval and --format-only cannot be both specified')

    print(f'Loading config: {args.config}')
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    mode = 'MMEngine/MMYOLO' if is_mmengine_style_cfg(cfg) else 'Legacy MMDet'
    print(f'Detected config mode: {mode}')

    overall_start = time.perf_counter()
    if is_mmengine_style_cfg(cfg):
        eval_mmengine_results(cfg, args)
    else:
        eval_legacy_results(cfg, args)
    overall_elapsed = time.perf_counter() - overall_start
    print(f'Total elapsed time: {format_seconds(overall_elapsed)}')


if __name__ == '__main__':
    main()
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import math

try:
    from mmengine.config import Config, DictAction
    from mmengine.evaluator import Evaluator
    from mmengine.fileio import load as mmengine_load
    from mmengine.registry import init_default_scope
    HAS_MMENGINE = True
except ImportError:
    HAS_MMENGINE = False
    import mmcv
    from mmcv import Config, DictAction


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metrics from pkl results.')
    parser.add_argument('config', help='config of the model')
    parser.add_argument('pkl_results', help='results in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='format output results without evaluation')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, e.g. "bbox", "segm", "proposal"')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override settings in used config, key=value format')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for dataset/evaluator')
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=200,
        help='chunk size for mmengine offline evaluation')
    args = parser.parse_args()
    return args


def is_mmengine_style_cfg(cfg):
    return ('test_dataloader' in cfg or cfg.get('default_scope', None) == 'mmyolo')


def build_dataset_meta(cfg):
    classes = None
    if cfg.get('metainfo', None) and cfg.metainfo.get('classes', None):
        classes = tuple(cfg.metainfo['classes'])
    elif cfg.get('class_name', None):
        classes = tuple(cfg.class_name)

    if classes is None:
        return None
    return {'classes': classes}


def eval_mmengine_results(cfg, args):
    if not HAS_MMENGINE:
        raise RuntimeError(
            'This config is MMEngine/MMYOLO-style, but mmengine is unavailable '
            'in the current environment.')

    print('[1/4] Initializing MMEngine scope...')
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    print(f'[2/4] Loading predictions from: {args.pkl_results}')
    outputs = mmengine_load(args.pkl_results)
    total = len(outputs)
    print(f'Loaded {total} predictions.')

    evaluator_cfg = copy.deepcopy(cfg.test_evaluator)
    kwargs = {} if args.eval_options is None else args.eval_options

    if args.eval:
        # Keep legacy behavior: --eval overrides config metric.
        evaluator_cfg['metric'] = args.eval if len(args.eval) > 1 else args.eval[0]

    if args.format_only:
        evaluator_cfg['format_only'] = True

    evaluator_cfg.update(kwargs)
    evaluator = Evaluator(evaluator_cfg)

    dataset_meta = build_dataset_meta(cfg)
    if dataset_meta is not None:
        evaluator.dataset_meta = dataset_meta
        print(f'Using {len(dataset_meta["classes"])} classes from config metainfo.')
    else:
        print('No class metainfo found in config; evaluator will infer defaults.')

    print('[3/4] Processing predictions for metric accumulation...')
    if total == 0:
        raise ValueError('No predictions found in pkl file.')

    chunk_size = max(1, args.chunk_size)
    num_chunks = math.ceil(total / chunk_size)
    for idx in range(num_chunks):
        start = idx * chunk_size
        end = min((idx + 1) * chunk_size, total)
        evaluator.process(data_samples=outputs[start:end], data_batch=None)
        print(f'  - processed chunk {idx + 1}/{num_chunks} '
              f'({end}/{total} samples)')

    print('[4/4] Computing final metrics...')
    metrics = evaluator.evaluate(total)
    print(metrics)


def eval_legacy_results(cfg, args):
    import mmcv
    from mmdet.datasets import build_dataset
    from mmdet.utils import replace_cfg_vals, update_data_root

    print('[1/4] Preparing legacy MMDet config...')
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True

    print('[2/4] Building dataset...')
    dataset = build_dataset(cfg.data.test)

    print(f'[3/4] Loading predictions from: {args.pkl_results}')
    outputs = mmcv.load(args.pkl_results)
    print(f'Loaded {len(outputs)} predictions.')

    kwargs = {} if args.eval_options is None else args.eval_options
    if args.format_only:
        print('[4/4] Formatting output results...')
        dataset.format_results(outputs, **kwargs)
        print('Formatting complete.')
    if args.eval:
        print('[4/4] Running evaluation...')
        eval_kwargs = cfg.get('evaluation', {}).copy()
        # Hard-code way to remove EvalHook args.
        for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule'
        ]:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        print(dataset.evaluate(outputs, **eval_kwargs))


def main():
    args = parse_args()

    assert args.eval or args.format_only, (
        'Please specify at least one operation with "--eval" or "--format-only".')
    if args.eval and args.format_only:
        raise ValueError('--eval and --format-only cannot be both specified')

    print(f'Loading config: {args.config}')
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    mode = 'MMEngine/MMYOLO' if is_mmengine_style_cfg(cfg) else 'Legacy MMDet'
    print(f'Detected config mode: {mode}')

    if is_mmengine_style_cfg(cfg):
        eval_mmengine_results(cfg, args)
    else:
        eval_legacy_results(cfg, args)


if __name__ == '__main__':
    main()
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy

try:
    from mmengine.config import Config, DictAction
    from mmengine.evaluator import Evaluator
    from mmengine.fileio import load as mmengine_load
    from mmengine.registry import init_default_scope
    HAS_MMENGINE = True
except ImportError:
    HAS_MMENGINE = False
    import mmcv
    from mmcv import Config, DictAction


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                     'results saved in pkl format')
    parser.add_argument('config', help='Config of the model')
    parser.add_argument('pkl_results', help='Results in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is '
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='Evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=200,
        help='chunk size for mmengine offline evaluation of large pkl results')
    args = parser.parse_args()
    return args


def is_mmengine_style_cfg(cfg):
    return ('test_dataloader' in cfg or cfg.get('default_scope', None) == 'mmyolo')


def build_dataset_meta(cfg):
    classes = None
    if cfg.get('metainfo', None) and cfg.metainfo.get('classes', None):
        classes = tuple(cfg.metainfo['classes'])
    elif cfg.get('class_name', None):
        classes = tuple(cfg.class_name)

    if classes is None:
        return None
    return {'classes': classes}


def eval_mmengine_results(cfg, args):
    if not HAS_MMENGINE:
        raise RuntimeError(
            'This config is MMEngine/MMYOLO-style, but mmengine is not '
            'available in current env.')

    init_default_scope(cfg.get('default_scope', 'mmdet'))
    outputs = mmengine_load(args.pkl_results)

    evaluator_cfg = copy.deepcopy(cfg.test_evaluator)
    kwargs = {} if args.eval_options is None else args.eval_options

    if args.eval:
        # Keep legacy behavior: --eval overrides config metric.
        evaluator_cfg['metric'] = args.eval if len(args.eval) > 1 else args.eval[0]

    if args.format_only:
        evaluator_cfg['format_only'] = True

    evaluator_cfg.update(kwargs)
    evaluator = Evaluator(evaluator_cfg)

    dataset_meta = build_dataset_meta(cfg)
    if dataset_meta is not None:
        evaluator.dataset_meta = dataset_meta

    metrics = evaluator.offline_evaluate(outputs, chunk_size=args.chunk_size)
    print(metrics)


def eval_legacy_results(cfg, args):
    import mmcv
    from mmdet.datasets import build_dataset
    from mmdet.utils import replace_cfg_vals, update_data_root

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    outputs = mmcv.load(args.pkl_results)

    kwargs = {} if args.eval_options is None else args.eval_options
    if args.format_only:
        dataset.format_results(outputs, **kwargs)
    if args.eval:
        eval_kwargs = cfg.get('evaluation', {}).copy()
        # hard-code way to remove EvalHook args
        for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule'
        ]:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        print(dataset.evaluate(outputs, **eval_kwargs))


def main():
    args = parse_args()

    assert args.eval or args.format_only, (
        'Please specify at least one operation (eval/format the results) with '
        'the argument "--eval", "--format-only"')
    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if is_mmengine_style_cfg(cfg):
        eval_mmengine_results(cfg, args)
    else:
        eval_legacy_results(cfg, args)


if __name__ == '__main__':
    main()
# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import mmcv
from mmcv import Config, DictAction

from mmdet.datasets import build_dataset
from mmdet.utils import replace_cfg_vals, update_data_root


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                     'results saved in pkl format')
    parser.add_argument('config', help='Config of the model')
    parser.add_argument('pkl_results', help='Results in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='Evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    assert args.eval or args.format_only, (
        'Please specify at least one operation (eval/format the results) with '
        'the argument "--eval", "--format-only"')
    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    outputs = mmcv.load(args.pkl_results)

    kwargs = {} if args.eval_options is None else args.eval_options
    if args.format_only:
        dataset.format_results(outputs, **kwargs)
    if args.eval:
        eval_kwargs = cfg.get('evaluation', {}).copy()
        # hard-code way to remove EvalHook args
        for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule'
        ]:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
