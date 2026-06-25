"""
scripts/eval_ks_standalone.py
------------------------------
Standalone KS evaluation script.  Loads a saved FNO1d checkpoint and runs
eval_ks() to produce noisy-validation and clean-test KS results.

Usage:
    python scripts/eval_ks_standalone.py \
        --prefix  baseline_ks_ns0.3_xl100_bs20_ts10_s42 \
        --modes   128 \
        --width   256 \
        --gpu     0 \
        --eval_lengths 100 1000 \
        --ks_data_val   ks_data_x_single_traj/ks_single_traj_expanded_10_10/ks_single_traj_val \
        --ks_data_test  ks_data_x_single_traj/ks_single_traj_expanded_10_10/ks_single_traj_test \
        --ks_noisy_eval_split test \
        --ks_clean_eval_init_noise 0.0 \
        --noisy_scale   0.3 \
        --output_folder ks_output_foulder

The checkpoint is expected at:
    saved_checkpoints/operators/{prefix}/{epochs-1:03d}
"""

import argparse
import os
import sys
import torch

current = os.path.dirname(os.path.realpath(__file__))
parent  = os.path.dirname(current)
sys.path.append(parent)
sys.path.append(current)

from models.fno_1d_new import FNO1d
from eval_scripts.eval_utils import load_operator
from eval_scripts.eval_ks import eval_ks


def parse_args():
    p = argparse.ArgumentParser(description='Standalone KS evaluator')
    p.add_argument('--prefix',        required=True,  type=str)
    p.add_argument('--modes',         default=128,    type=int)
    p.add_argument('--width',         default=256,    type=int)
    p.add_argument('--gpu',           default=0,      type=int)
    p.add_argument('--x_len',         default=1000,   type=int,
                   help='Backwards-compatible single rollout length. Ignored if --eval_lengths is set.')
    p.add_argument('--eval_lengths',  nargs='+',      type=int, default=None,
                   help='One or more rollout lengths, e.g. --eval_lengths 100 1000')
    p.add_argument('--epochs',        default=701,    type=int,
                   help='Total training epochs (checkpoint index = epochs-1)')
    p.add_argument('--ks_data_val',   default='ks_data_x_single_traj/ks_single_traj_expanded_10_10/ks_single_traj_val', type=str)
    p.add_argument('--ks_data_test',  default='ks_data_x_single_traj/ks_single_traj_expanded_10_10/ks_single_traj_test', type=str)
    p.add_argument('--ks_noisy_eval_split', default='test', choices=['test', 'validation'], type=str,
                   help='Split used for noisy evaluation. Default is test.')
    p.add_argument('--ks_clean_eval_init_noise', default=0.0, type=float,
                   help='Noise level used for clean-eval initial conditions. Default 0.0.')
    p.add_argument('--noisy_scale',   default=0.0,    type=float)
    p.add_argument('--skip_noisy_eval', action='store_true',
                   help='Skip the noisy branch and only run clean KS evaluation.')
    p.add_argument('--skip_clean_eval', action='store_true',
                   help='Skip the clean branch and only run noisy KS evaluation.')
    p.add_argument('--x_len_train',   default=100,    type=int)
    p.add_argument('--output_folder', default='ks_output_foulder',   type=str)
    p.add_argument('--ks_N',          default=256,    type=int)
    p.add_argument('--ks_plot_samples', default=3,   type=int)
    p.add_argument('--operators_path',
                   default='saved_checkpoints/operators',
                   type=str,
                   help='Root path where operator checkpoints are stored')
    return p.parse_args()


def main():
    args = parse_args()
    eval_lengths = args.eval_lengths if args.eval_lengths else [args.x_len]
    args.x_len = args.x_len_train
    args.ks_eval_lengths = eval_lengths

    # ── Build operator ────────────────────────────────────────────────────────
    operator = FNO1d(modes=args.modes, width=args.width).cuda(args.gpu)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ep = args.epochs - 1
    ckpt_path = os.path.join(args.operators_path, args.prefix, f'{ep:03d}')
    if not os.path.exists(ckpt_path):
        print(f'[ERROR] Checkpoint not found: {ckpt_path}')
        sys.exit(1)

    operator = load_operator(operator, saved_pth=ckpt_path)
    operator.eval()

    # ── Output path ───────────────────────────────────────────────────────────
    output_path = os.path.join(args.output_folder, args.prefix)
    os.makedirs(output_path, exist_ok=True)

    # ── Run evaluation ────────────────────────────────────────────────────────
    for eval_len in eval_lengths:
        if not args.skip_noisy_eval:
            eval_ks(
                operator,
                args,
                noisy_scale=args.noisy_scale,
                x_len=eval_len,
                output_path=output_path,
            )
        if not args.skip_clean_eval:
            eval_ks(
                operator,
                args,
                noisy_scale=0,
                x_len=eval_len,
                output_path=output_path,
            )

    print(f'\nResults written to {os.path.join(output_path, "eval_noisy_trainval_clean_test")}')


if __name__ == '__main__':
    main()
