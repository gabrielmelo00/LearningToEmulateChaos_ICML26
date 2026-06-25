"""
scripts/summary_ks.py
---------------------
Aggregate KS benchmark results across seeds, batch sizes, rollouts, and
evaluation noise levels.

Examples
--------
python scripts/summary_ks.py --output_folder ks_output_foulder
python scripts/summary_ks.py --output_folder ks_output_foulder --split clean --rollout 100 1000
python scripts/summary_ks.py --output_folder ks_output_foulder --method wgan --split noisy --train_noise 0.3
"""

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


METHOD_MAP = {
    'baseline': 'Baseline (MSE)',
    'ot_fixed': 'OT Fixed',
    'sinkhorn': 'Sinkhorn',
    'wgan': 'WGAN-OT',
}
METHOD_ORDER = ['baseline', 'ot_fixed', 'sinkhorn', 'wgan']
NUM_TOKEN = r'(?:[0-9.eE+-]+|nan|inf|-inf)'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', default='ks_output_foulder', type=str)
    parser.add_argument('--split', choices=['all', 'clean', 'noisy', 'noisy_test', 'noisy_val'], default='all')
    parser.add_argument('--method', choices=METHOD_ORDER, default=None)
    parser.add_argument('--rollout', nargs='*', type=int, default=None)
    parser.add_argument('--batch_sizes', nargs='*', type=int, default=None)
    parser.add_argument('--train_noise', type=str, default=None,
                        help='Filter by training noise encoded in the run prefix, e.g. 0.3')
    parser.add_argument('--eval_noise', type=str, default=None,
                        help='Filter by evaluation noise read from the result file, e.g. 0, 0.1, 0.3')
    return parser.parse_args()


def split_method(folder_name):
    for method in ['ot_fixed', 'baseline', 'sinkhorn', 'wgan']:
        prefix = f'{method}_ks_'
        if folder_name.startswith(prefix):
            return method, folder_name[len(prefix):]
    return None, None


def parse_folder_metadata(folder_name):
    method, rest = split_method(folder_name)
    if method is None:
        return None

    metadata = {
        'method': method,
        'folder': folder_name,
    }
    for token in rest.split('_'):
        if token.startswith('ns'):
            metadata['train_noise'] = token[2:]
        elif token.startswith('xl'):
            metadata['x_len_train'] = int(token[2:])
        elif token.startswith('bs'):
            metadata['batch_size'] = int(token[2:])
        elif token.startswith('ts'):
            metadata['train_size'] = int(token[2:])
        elif token.startswith('sdim'):
            metadata['summary_dim'] = int(token[4:])
        elif token.startswith('steps'):
            metadata['critic_steps'] = int(token[5:])
        elif token.startswith('clip'):
            metadata['clip'] = token[4:]
        elif token.startswith('s') and token[1:].isdigit():
            metadata['seed'] = int(token[1:])

    required = ['train_noise', 'x_len_train', 'batch_size', 'train_size', 'seed']
    if any(key not in metadata for key in required):
        return None
    return metadata


def parse_result_text(text, default_split):
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    eval_noise = None
    rollout = None
    if lines and lines[0].startswith('noise '):
        m = re.search(r'noise ([0-9.eE+-]+) with eval length ([0-9]+) training length ([0-9]+)', lines[0])
        if m:
            eval_noise = m.group(1)
            rollout = int(m.group(2))
    if rollout is None:
        m = re.search(r'rollout_len=([0-9]+)', text)
        if m:
            rollout = int(m.group(1))
    if eval_noise is None:
        eval_noise = '0'

    rmse_match = re.search(
        rf"mse_\['rMSE', ({NUM_TOKEN}), '50 percentile', ({NUM_TOKEN}), ({NUM_TOKEN})\]",
        text,
        flags=re.IGNORECASE,
    )
    if rmse_match is not None:
        rmse_q50 = float(rmse_match.group(2))
        spec_q50 = float(
            re.search(
                rf'spectrum distance:\[[^,\]]+, array\(\[\[?50\]?\]\), ({NUM_TOKEN}),',
                text,
                flags=re.IGNORECASE,
            ).group(1)
        )
        l1_q50 = float(
            re.search(
                rf'l1_3d_score: \[[^,\]]+, [^,\]]+, .+?, ({NUM_TOKEN}),',
                text,
                flags=re.IGNORECASE,
            ).group(1)
        )
        return {
            'split': default_split,
            'eval_noise': eval_noise,
            'rollout': rollout,
            'rmse_q50': rmse_q50,
            'spec_q50': spec_q50,
            'l1_q50': l1_q50,
        }

    rmse_match = re.search(rf'One-step relative L2 \(RMSE\):\s*({NUM_TOKEN})', text, flags=re.IGNORECASE)
    if rmse_match is not None:
        spec_q50 = float(
            re.search(
                rf'Spectrum distance\s+q25/q50/q75:\s*{NUM_TOKEN} / ({NUM_TOKEN})',
                text,
                flags=re.IGNORECASE,
            ).group(1)
        )
        l1_q50 = float(
            re.search(
                rf'L1 score\s+q25/q50/q75:\s*{NUM_TOKEN} / ({NUM_TOKEN})',
                text,
                flags=re.IGNORECASE,
            ).group(1)
        )
        return {
            'split': default_split,
            'eval_noise': eval_noise,
            'rollout': rollout,
            'rmse_q50': float(rmse_match.group(1)),
            'spec_q50': spec_q50,
            'l1_q50': l1_q50,
        }

    return None


def discover_result_paths(exp_dir):
    result_paths = []
    eval_root = exp_dir / 'eval_noisy_trainval_clean_test'

    if eval_root.is_dir():
        rollout_dirs = sorted(
            d for d in eval_root.iterdir()
            if d.is_dir() and d.name.startswith('rollout_')
        )
        if rollout_dirs:
            for rollout_dir in rollout_dirs:
                clean_path = rollout_dir / 'test_on_clean_data' / 'Results_test_on_clean_data.txt'
                if clean_path.exists():
                    result_paths.append((clean_path, 'clean'))
                for noisy_test_dir in sorted(rollout_dir.glob('test_on_noise_data*')):
                    noisy_test_path = noisy_test_dir / 'Results_test_on_noise_data.txt'
                    if noisy_test_path.exists():
                        result_paths.append((noisy_test_path, 'noisy_test'))
                for noisy_val_dir in sorted(rollout_dir.glob('validation_on_noise_data*')):
                    noisy_val_path = noisy_val_dir / 'Results_validation_on_noise_data.txt'
                    if noisy_val_path.exists():
                        result_paths.append((noisy_val_path, 'noisy_val'))
            return result_paths

        clean_path = eval_root / 'test_on_clean_data' / 'Results_test_on_clean_data.txt'
        if clean_path.exists():
            result_paths.append((clean_path, 'clean'))
        for noisy_test_dir in sorted(eval_root.glob('test_on_noise_data*')):
            noisy_test_path = noisy_test_dir / 'Results_test_on_noise_data.txt'
            if noisy_test_path.exists():
                result_paths.append((noisy_test_path, 'noisy_test'))
        for noisy_val_dir in sorted(eval_root.glob('validation_on_noise_data*')):
            noisy_val_path = noisy_val_dir / 'Results_validation_on_noise_data.txt'
            if noisy_val_path.exists():
                result_paths.append((noisy_val_path, 'noisy_val'))
        if result_paths:
            return result_paths

    for legacy_path in sorted(exp_dir.glob('Results_ks_test_rollout_*.txt')):
        result_paths.append((legacy_path, 'clean'))
    if not result_paths:
        legacy_path = exp_dir / 'Results_ks_test.txt'
        if legacy_path.exists():
            result_paths.append((legacy_path, 'clean'))
    return result_paths


def as_float(text_value):
    return float(str(text_value))


def keep_record(record, args):
    if args.split == 'clean' and record['split'] != 'clean':
        return False
    if args.split == 'noisy' and record['split'] not in {'noisy_test', 'noisy_val'}:
        return False
    if args.split == 'noisy_test' and record['split'] != 'noisy_test':
        return False
    if args.split == 'noisy_val' and record['split'] != 'noisy_val':
        return False
    if args.method is not None and record['method'] != args.method:
        return False
    if args.rollout is not None and record['rollout'] not in args.rollout:
        return False
    if args.batch_sizes is not None and record['batch_size'] not in args.batch_sizes:
        return False
    if args.train_noise is not None and str(record['train_noise']) != str(args.train_noise):
        return False
    if args.eval_noise is not None and str(record['eval_noise']) != str(args.eval_noise):
        return False
    return True


def result_key(record):
    split_order = {
        'clean': 0,
        'noisy_test': 1,
        'noisy_val': 2,
    }
    return (
        METHOD_ORDER.index(record['method']),
        split_order.get(record['split'], 99),
        as_float(record['train_noise']),
        as_float(record['eval_noise']),
        record['batch_size'],
        record['rollout'],
    )


def main():
    args = parse_args()
    output_root = Path(args.output_folder)
    if not output_root.exists():
        raise FileNotFoundError(f'Output folder not found: {output_root}')

    grouped = defaultdict(list)
    missing = []

    for folder_name in sorted(os.listdir(output_root)):
        metadata = parse_folder_metadata(folder_name)
        if metadata is None:
            continue

        exp_dir = output_root / folder_name
        result_paths = discover_result_paths(exp_dir)
        if not result_paths:
            missing.append(folder_name)
            continue

        for result_path, default_split in result_paths:
            with open(result_path, encoding='utf-8') as f:
                parsed = parse_result_text(f.read(), default_split=default_split)
            if parsed is None or parsed['rollout'] is None:
                continue

            record = {**metadata, **parsed}
            if not keep_record(record, args):
                continue
            key = (
                record['method'],
                record['split'],
                record['train_noise'],
                record['eval_noise'],
                record['batch_size'],
                record['rollout'],
            )
            grouped[key].append(record)

    if missing:
        print(f'[INFO] {len(missing)} experiment folders have no parseable KS results yet:')
        for folder in missing:
            print(f'  {folder}')
        print()

    rows = []
    for key, records in grouped.items():
        metrics = np.array([[r['rmse_q50'], r['spec_q50'], r['l1_q50']] for r in records], dtype=float)
        n = len(metrics)
        means = metrics.mean(axis=0)
        stds = metrics.std(axis=0, ddof=1) if n > 1 else np.zeros(3)
        spec_var = metrics[:, 1].var(ddof=1) if n > 1 else 0.0
        row = {
            'method': key[0],
            'split': key[1],
            'train_noise': key[2],
            'eval_noise': key[3],
            'batch_size': key[4],
            'rollout': key[5],
            'rmse_mean': means[0],
            'rmse_std': stds[0],
            'spec_mean': means[1],
            'spec_std': stds[1],
            'spec_var': spec_var,
            'l1_mean': means[2],
            'l1_std': stds[2],
            'n_seeds': n,
        }
        rows.append(row)

    rows.sort(key=result_key)

    if not rows:
        print('No rows matched the requested filters.')
        return

    header = (
        f"{'Method':<16} {'Split':<10} {'TrainNS':>7} {'EvalNS':>7} "
        f"{'BS':>4} {'Roll':>6} {'RMSE q50':>18} {'Spec q50':>18} "
        f"{'Spec Var':>12} {'L1 q50':>18} {'N':>4}"
    )
    print(header)
    print('-' * len(header))
    for row in rows:
        print(
            f"{METHOD_MAP.get(row['method'], row['method']):<16} "
            f"{row['split']:<10} "
            f"{row['train_noise']:>7} "
            f"{row['eval_noise']:>7} "
            f"{row['batch_size']:>4} "
            f"{row['rollout']:>6} "
            f"{row['rmse_mean']:.4f}±{row['rmse_std']:.4f} "
            f"{row['spec_mean']:.4f}±{row['spec_std']:.4f} "
            f"{row['spec_var']:.6f} "
            f"{row['l1_mean']:.4f}±{row['l1_std']:.4f} "
            f"{row['n_seeds']:>4}"
        )


if __name__ == '__main__':
    main()
