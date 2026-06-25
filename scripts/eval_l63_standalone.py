"""
Standalone L63 evaluator.

Loads a saved operator checkpoint and runs noisy + clean evaluation.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
sys.path.append(current)

from eval_scripts.eval_l63 import eval_l63
from eval_scripts.eval_utils import load_operator
from models.mlp_l63 import L63MLP


def parse_args():
    p = argparse.ArgumentParser(description='Standalone L63 evaluator')
    p.add_argument('--prefix', required=True, type=str)
    p.add_argument('--modes', default=2, type=int)
    p.add_argument('--width', default=64, type=int)
    p.add_argument('--gpu', default=0, type=int)
    p.add_argument('--epochs', default=701, type=int,
                   help='Training epochs (checkpoint index is epochs-1)')

    p.add_argument('--x_len_train', default=300, type=int)
    p.add_argument('--x_len', default=1000, type=int,
                   help='Fallback single rollout length when --eval_lengths is not set')
    p.add_argument('--eval_lengths', nargs='+', type=int, default=None)

    p.add_argument('--l63_data_val', default='l63_data_x/l63_data_val', type=str)
    p.add_argument('--l63_data_test', default='l63_data_x/l63_data_test', type=str)
    p.add_argument('--l63_noisy_eval_split', default='test', choices=['test', 'validation'], type=str)
    p.add_argument('--l63_clean_eval_init_noise', default=0.0, type=float)
    p.add_argument('--l63_plot_samples', default=3, type=int)

    p.add_argument('--noisy_scale', default=0.3, type=float)
    p.add_argument('--skip_noisy_eval', action='store_true')
    p.add_argument('--skip_clean_eval', action='store_true')

    p.add_argument('--output_folder', default='l63_output_folder', type=str)
    p.add_argument('--operators_path', default='saved_checkpoints/operators', type=str)
    return p.parse_args()


def main():
    args = parse_args()
    eval_lengths = args.eval_lengths if args.eval_lengths else [args.x_len]
    args.x_len = args.x_len_train
    args.l63_eval_lengths = eval_lengths

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device(f'cuda:{args.gpu % torch.cuda.device_count()}')
    else:
        device = torch.device('cpu')
        print('CUDA is not available. Running L63 standalone evaluation on CPU.')
    args.gpu = device

    operator = L63MLP(hidden_dim=args.width).to(device)

    ep = args.epochs - 1
    ckpt_path = os.path.join(args.operators_path, args.prefix, f'{ep:03d}')
    if not os.path.exists(ckpt_path):
        print(f'[ERROR] Checkpoint not found: {ckpt_path}')
        sys.exit(1)

    operator = load_operator(operator, saved_pth=ckpt_path)
    operator.eval()

    output_path = os.path.join(args.output_folder, args.prefix)
    os.makedirs(output_path, exist_ok=True)

    for eval_len in eval_lengths:
        if not args.skip_noisy_eval:
            eval_l63(
                operator,
                args,
                noisy_scale=args.noisy_scale,
                x_len=eval_len,
                output_path=output_path,
            )
        if not args.skip_clean_eval:
            eval_l63(
                operator,
                args,
                noisy_scale=0.0,
                x_len=eval_len,
                output_path=output_path,
            )

    print(f'\nResults written to {os.path.join(output_path, "eval_noisy_trainval_clean_test")}')


if __name__ == '__main__':
    main()
