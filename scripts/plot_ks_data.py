"""
scripts/plot_ks_data.py
-----------------------
Quick visualization of generated KS single-trajectory data.
Uses plot_3col from train_utils to produce x-t heatmaps.

Usage (from project root):
    .venv/bin/python scripts/plot_ks_data.py
    .venv/bin/python scripts/plot_ks_data.py --folder ks_data_x_single_traj/ks_single_traj_train --n_samples 5
"""

import argparse
import os
import sys

import numpy as np
import torch

current = os.path.dirname(os.path.realpath(__file__))
parent  = os.path.dirname(current)
sys.path.append(parent)

from scripts.train_utils import plot_3col

parser = argparse.ArgumentParser()
parser.add_argument('--folder',    default='ks_data_x_single_traj/ks_single_traj_train', type=str)
parser.add_argument('--n_samples', default=3,   type=int, help='Number of trajectories to plot')
parser.add_argument('--out_dir',   default='plots/ks_data', type=str)
args = parser.parse_args()

data_folder = os.path.join(parent, args.folder)
os.makedirs(args.out_dir, exist_ok=True)

files = sorted([f for f in os.listdir(data_folder) if f.endswith('.pth') and f != 'training_params.pth'])
files = files[:args.n_samples]

for fname in files:
    d    = torch.load(os.path.join(data_folder, fname), map_location='cpu', weights_only=False)
    traj = d['1'].numpy()   # (T, N)
    stem = os.path.splitext(fname)[0]

    # plot_3col expects two panels; show first half vs second half of the trajectory
    mid  = traj.shape[0] // 2
    plot_3col(
        [traj[:mid], f'{stem} — first half'],
        [traj[mid:mid*2], f'{stem} — second half'],
        im=os.path.join(args.out_dir, stem),
    )
    print(f'Saved {args.out_dir}/{stem}.png  (T={traj.shape[0]}, N={traj.shape[1]})')
