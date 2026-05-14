# Copyright (c) OpenMMLab. All rights reserved.
"""Plot curves from MMEngine / MMYOLO scalar logs (JSONL, e.g. vis_data/scalars.json).

MMDet-style ``*.log.json`` files are handled by ``analyze_logs.py``; this script
targets one-json-object-per-line logs written under ``work_dirs/.../vis_data/``.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot metrics from MMEngine scalars.json (JSONL)')
    parser.add_argument(
        'scalars_json',
        type=str,
        nargs='+',
        help='path(s) to JSONL scalar log(s), e.g. '
        'work_dirs/exp/TIMESTAMP/vis_data/scalars.json')
    parser.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['loss'],
        help='metric name(s) stored as keys on each JSON line')
    parser.add_argument(
        '--x-key',
        type=str,
        default='step',
        help='x-axis key (default: step). Common alternatives: iter')
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='output image path (.png, .pdf, etc.)')
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend entries; default is '
        '"{basename}_{key}" for each file and key')
    parser.add_argument(
        '--backend',
        type=str,
        default=None,
        help='matplotlib backend (e.g. Agg for headless)')
    parser.add_argument(
        '--style',
        type=str,
        default='dark',
        help='seaborn style preset')
    return parser.parse_args()


def load_series(path, x_key, y_key):
    xs, ys = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if x_key not in d or y_key not in d:
                continue
            v_x, v_y = d[x_key], d[y_key]
            if isinstance(v_x, (int, float)) and isinstance(v_y, (int, float)):
                xs.append(v_x)
                ys.append(v_y)
    return xs, ys


def main():
    args = parse_args()
    if args.backend is not None:
        plt.switch_backend(args.backend)
    try:
        import seaborn as sns
        sns.set_style(args.style)
    except ImportError:
        pass

    paths = args.scalars_json
    keys = args.keys
    n_series = len(paths) * len(keys)
    if args.legend is not None:
        if len(args.legend) != n_series:
            raise ValueError(
                f'Expected {n_series} legend entries '
                f'({len(paths)} files x {len(keys)} keys), '
                f'got {len(args.legend)}')
        legend = args.legend
    else:
        legend = []
        for p in paths:
            tag = Path(p).stem
            for k in keys:
                legend.append(f'{tag}_{k}')

    plt.figure(figsize=(10, 4))
    for i, path in enumerate(paths):
        for j, key in enumerate(keys):
            xs, ys = load_series(path, args.x_key, key)
            if not xs:
                raise RuntimeError(
                    f'No points with both x_key={args.x_key!r} and '
                    f'y_key={key!r} in {path}')
            idx = i * len(keys) + j
            plt.plot(
                xs, ys, linewidth=0.5, label=legend[idx])

    plt.xlabel(args.x_key)
    if len(keys) == 1:
        plt.ylabel(keys[0])
    else:
        plt.ylabel('value')
    if args.title:
        plt.title(args.title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f'saved curve to: {args.out}')


if __name__ == '__main__':
    main()
