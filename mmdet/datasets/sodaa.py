import itertools
import json
import logging
import os
import os.path as osp
import tempfile
import time

import torch
import mmcv
import numpy as np
from mmcv.utils import print_log
from pycocotools.coco import COCO
from terminaltables import AsciiTable
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

from mmdet.core import eval_recalls
from .builder import DATASETS
from .custom import CustomDataset
from .sodad_eval.sodadeval import SODADeval
from mmcv.ops.nms import nms


@DATASETS.register_module(force=True)
class SODAADataset(CustomDataset):
    # 9 foreground categories (excluding 'ignore')
    CLASSES = ('airplane', 'helicopter', 'small-vehicle', 'large-vehicle',
               'ship', 'container', 'storage-tank', 'swimming-pool', 'windmill')

    def __init__(self, ori_ann_file, **kwargs):
        super(SODAADataset, self).__init__(**kwargs)
        # Build COCO object from original (un-split) annotations for evaluation
        ori_dataset = self._load_ann_dir(ori_ann_file)
        self.ori_coco = COCO()
        self.ori_coco.dataset = ori_dataset
        self.ori_coco.createIndex()
        self.ori_img_ids = self.ori_coco.getImgIds()

    def _load_ann_dir(self, ann_dir):
        """Merge all per-image JSON files in ann_dir into a single dataset dict.

        SODA-A stores one JSON file per image. Each file has:
            images: dict (single image info)
            annotations: list of annotation dicts with 'poly' field
            categories: list of category dicts
        """
        all_images = []
        all_annotations = []
        categories = None

        ann_id_counter = 1
        img_id_counter = 1
        json_files = sorted(f for f in os.listdir(ann_dir) if f.endswith('.json'))
        for json_file in json_files:
            with open(osp.join(ann_dir, json_file)) as f:
                data = json.load(f)

            if categories is None:
                categories = data.get('categories', [])

            is_split_fmt = 'images' not in data  # split format has image info at top level

            if is_split_fmt:
                # Split format produced by sodaa_split.py
                img_info = {
                    'id': img_id_counter,
                    'file_name': data['filename'],
                    'filename': data['filename'],
                    'height': data['height'],
                    'width': data['width'],
                    'x_start': data.get('x_start', 0),
                    'y_start': data.get('y_start', 0),
                    'ori_id': int(data['ori_id']) if 'ori_id' in data else img_id_counter,
                }
                img_id = img_id_counter
                img_id_counter += 1
                all_images.append(img_info)
            else:
                # Original per-image format
                images = data['images']
                if isinstance(images, dict):
                    images = [images]
                img_info = dict(images[0])
                img_info.setdefault('filename', img_info.get('file_name', ''))
                img_info.setdefault('x_start', 0)
                img_info.setdefault('y_start', 0)
                img_info.setdefault('ori_id', img_info['id'])
                img_id = img_info['id']
                all_images.append(img_info)

            for ann in data.get('annotations', []):
                ann = dict(ann)
                ann['id'] = ann_id_counter
                ann_id_counter += 1
                ann['image_id'] = img_id
                # Normalise category field (split format uses 'cat_id')
                if 'category_id' not in ann and 'cat_id' in ann:
                    ann['category_id'] = ann['cat_id']
                # Convert oriented poly to axis-aligned bbox (COCO xywh format)
                if 'bbox' not in ann and 'poly' in ann:
                    pts = np.array(ann['poly'], dtype=np.float32).reshape(-1, 2)
                    x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
                    x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
                    ann['bbox'] = [x1, y1, x2 - x1, y2 - y1]
                    if 'area' not in ann or not ann['area'] or ann['area'] <= 0:
                        ann['area'] = float((x2 - x1) * (y2 - y1))
                if 'iscrowd' not in ann:
                    ann['iscrowd'] = 0
                all_annotations.append(ann)

        return {
            'images': all_images,
            'annotations': all_annotations,
            'categories': categories or [],
        }

    def load_annotations(self, ann_file):
        """Load annotation from a directory of per-image JSON files."""
        dataset = self._load_ann_dir(ann_file)

        self.coco = COCO()
        self.coco.dataset = dataset
        self.coco.createIndex()

        self.cat_ids = self.coco.getCatIds(catNms=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.getImgIds()

        data_infos = []
        for img_id in self.img_ids:
            info = self.coco.loadImgs([img_id])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos

    def get_ann_info(self, idx):
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.data_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    @staticmethod
    def _poly_to_xyxy(poly):
        """Convert 4-point polygon (8 floats) to axis-aligned [x1, y1, x2, y2]."""
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        return [float(pts[:, 0].min()), float(pts[:, 1].min()),
                float(pts[:, 0].max()), float(pts[:, 1].max())]

    def _parse_ann_info(self, img_info, ann_info):
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []

        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            if ann['category_id'] not in self.cat_ids:
                continue

            # Use pre-converted bbox [x, y, w, h] stored during load
            x, y, w, h = ann['bbox']
            x1, y1, x2, y2 = x, y, x + w, y + h

            if ann.get('area', w * h) <= 0 or w < 1 or h < 1:
                continue

            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append([x1, y1, x2, y2])
            else:
                gt_bboxes.append([x1, y1, x2, y2])
                gt_labels.append(self.cat2label[ann['category_id']])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        return dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
        )

    def xyxy2xywh(self, bbox):
        _bbox = bbox.tolist()
        return [_bbox[0], _bbox[1], _bbox[2] - _bbox[0], _bbox[3] - _bbox[1]]

    def _det2json(self, results, score_thr=0.1, max_dets_per_img=3000):
        json_results = []
        for idx in range(len(self.ori_img_ids)):
            img_id = self.ori_img_ids[idx]
            result = results[idx]
            img_dets = []
            for label in range(len(result)):
                bboxes = result[label]
                # Filter low-confidence detections
                if len(bboxes) > 0 and score_thr > 0:
                    bboxes = bboxes[bboxes[:, 4] >= score_thr]
                for i in range(bboxes.shape[0]):
                    img_dets.append(dict(
                        image_id=img_id,
                        bbox=self.xyxy2xywh(bboxes[i]),
                        score=float(bboxes[i][4]),
                        category_id=self.cat_ids[label],
                    ))
            # Cap total detections per image to avoid evaluation slowdown
            if len(img_dets) > max_dets_per_img:
                img_dets.sort(key=lambda x: -x['score'])
                img_dets = img_dets[:max_dets_per_img]
            json_results.extend(img_dets)
        return json_results

    def _proposal2json(self, results):
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def results2json(self, results, outfile_prefix):
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def translate(self, bboxes, x, y):
        dim = bboxes.shape[-1]
        return bboxes + np.array([x, y] * int(dim / 2), dtype=np.float32)

    def merge_dets(self, results, with_merge=True, nms_iou_thr=0.5,
                   nproc=10, save_dir=None, **kwargs):
        """Merge per-patch detection results to whole-image results.

        When with_merge=False (no image splitting), results are returned as-is
        paired with their original image IDs.
        """
        if not with_merge:
            results = [(data_info['id'], result)
                       for data_info, result in zip(self.data_infos, results)]
            return results

        print('\n>>> Merge detected results of patch for whole image evaluating...')
        start_time = time.time()
        collector = defaultdict(list)

        total = len(self.data_infos)
        for i, (data_info, result) in enumerate(zip(self.data_infos, results)):
            if i % 5000 == 0:
                print(f'  Collecting patches: {i}/{total}')
            x_start = data_info.get('x_start', 0)
            y_start = data_info.get('y_start', 0)
            new_result = []
            for j, res in enumerate(result):
                bboxes, scores = res[:, :-1], res[:, [-1]]
                bboxes = self.translate(bboxes, x_start, y_start)
                labels = np.zeros((bboxes.shape[0], 1)) + j
                new_result.append(np.concatenate([labels, bboxes, scores], axis=1))

            new_result = np.concatenate(new_result, axis=0)
            ori_id = data_info.get('ori_id', data_info['id'])
            collector[ori_id].append(new_result)

        print(f'  Collecting patches: {total}/{total}')
        print(f'  Running NMS across {len(collector)} original images...')
        merge_func = partial(_merge_func, CLASSES=self.CLASSES, iou_thr=nms_iou_thr)
        if nproc > 1:
            pool = Pool(nproc)
            merged_results = pool.map(merge_func, list(collector.items()))
            pool.close()
        else:
            merged_results = list(map(merge_func, list(collector.items())))

        stop_time = time.time()
        print('Merge results completed, it costs %.1f seconds.' % (stop_time - start_time))
        return merged_results

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.getAnnIds(imgIds=self.img_ids[i])
            ann_info = self.coco.loadAnns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann.get('iscrowd', False):
                    continue
                x, y, w, h = ann['bbox']
                bboxes.append([x, y, x + w, y + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        return recalls.mean(axis=1)

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=True,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None,
                 with_merge=False):
        """Evaluate detection results using the SODA evaluation protocol."""
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)

        merged_results = self.merge_dets(
            results=results,
            with_merge=with_merge,
            nms_iou_thr=0.5,
            nproc=8,
        )

        img_ids = [r[0] for r in merged_results]
        results = [r[1] for r in merged_results]

        # Sort to match ori_img_ids order
        empty_result = [np.empty((0, 5), dtype=np.float32) for _ in self.CLASSES]
        sort_results = []
        for ori_img_id in self.ori_img_ids:
            if ori_img_id in img_ids:
                sort_results.append(results[img_ids.index(ori_img_id)])
            else:
                sort_results.append(empty_result)
        results = sort_results

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = {}
        cocoGt = self.ori_coco

        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(results, proposal_nums, iou_thrs, logger='silent')
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                continue

            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')

            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            SODAAEval = SODADeval(cocoGt, cocoDt, iou_type)
            SODAAEval.params.catIds = self.cat_ids
            SODAAEval.params.imgIds = self.ori_img_ids
            SODAAEval.params.maxDets = list(proposal_nums)
            SODAAEval.params.iouThrs = iou_thrs

            tod_metric_names = {
                'AP': 0, 'AP_50': 1, 'AP_75': 2,
                'AP_eS': 3, 'AP_rS': 4, 'AP_gS': 5, 'AP_Normal': 6,
                'AR@100': 7, 'AR@300': 8, 'AR@1000': 9,
                'AR_eS@1000': 10, 'AR_rS@1000': 11,
                'AR_gS@1000': 12, 'AR_Normal@1000': 13,
            }

            if metric == 'proposal':
                SODAAEval.params.useCats = 0
            SODAAEval.evaluate()
            SODAAEval.accumulate()
            SODAAEval.summarize()

            if classwise:
                precisions = SODAAEval.eval['precision']
                assert len(self.cat_ids) == precisions.shape[2]
                results_per_category = []
                for idx, catId in enumerate(self.cat_ids):
                    nm = self.ori_coco.loadCats(catId)[0]
                    precision = precisions[:, :, idx, 0, -1]
                    precision = precision[precision > -1]
                    ap = float(np.mean(precision)) if precision.size else float('nan')
                    results_per_category.append((f'{nm["name"]}', f'{ap:0.3f}'))

                num_columns = min(6, len(results_per_category) * 2)
                results_flatten = list(itertools.chain(*results_per_category))
                headers = ['category', 'AP'] * (num_columns // 2)
                results_2d = itertools.zip_longest(
                    *[results_flatten[i::num_columns] for i in range(num_columns)])
                table_data = [headers] + [r for r in results_2d]
                table = AsciiTable(table_data)
                print_log('\n' + table.table, logger=logger)

            if metric_items is None:
                metric_items = ['AP', 'AP_50', 'AP_75', 'AP_eS', 'AP_rS', 'AP_gS', 'AP_Normal']

            for metric_item in metric_items:
                key = f'{metric}_{metric_item}'
                val = float(f'{SODAAEval.stats[tod_metric_names[metric_item]]:.3f}')
                eval_results[key] = val

            ap = SODAAEval.stats[:7]
            eval_results[f'{metric}_mAP_copypaste'] = (
                f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} '
                f'{ap[3]:.3f} {ap[4]:.3f} {ap[5]:.3f} {ap[6]:.3f}'
            )

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results


def _merge_func(info, CLASSES, iou_thr):
    img_id, label_dets = info
    label_dets = np.concatenate(label_dets, axis=0)
    labels, dets = label_dets[:, 0], label_dets[:, 1:]

    ori_img_results = []
    for i in range(len(CLASSES)):
        cls_dets = dets[labels == i]
        if len(cls_dets) == 0:
            ori_img_results.append(np.empty((0, dets.shape[1]), dtype=np.float32))
            continue
        bboxes = torch.from_numpy(cls_dets[:, :-1]).to(torch.float32).contiguous()
        scores = torch.from_numpy(cls_dets[:, -1]).to(torch.float32).contiguous()
        results, _ = nms(bboxes, scores, iou_thr)
        ori_img_results.append(results.numpy())
    return img_id, ori_img_results
