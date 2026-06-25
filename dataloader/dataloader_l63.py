"""
PyTorch datasets for Lorenz-63 experiments.

Dataset file format (per sample):
    {'0': params (3,), '1': traj (T, 3), 'ic': optional (3,)}

Noisy data are stored as sibling files with suffix:
    000123_noise_0.30.pth
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)


# -----------------------------------------------------------------------------
# Crop utilities
# -----------------------------------------------------------------------------

def crop(traj: torch.Tensor, start_idx: torch.Tensor, crop_len: int) -> torch.Tensor:
    assert start_idx.shape[0] == traj.shape[0]
    return torch.stack([traj[i, start_idx[i]: start_idx[i] + crop_len] for i in range(traj.shape[0])])


@torch.no_grad()
def random_cropping(traj: torch.Tensor, crop_len: int) -> torch.Tensor:
    t_len = traj.shape[1]
    assert t_len >= crop_len

    # Keep the same "skip transient" spirit as existing loaders, while still
    # working when trajectories are shorter than this margin.
    min_start = min(50, max(0, t_len - crop_len))
    max_start = t_len - crop_len
    if max_start <= min_start:
        start_idx = torch.zeros((traj.shape[0],), dtype=torch.long)
    else:
        start_idx = torch.randint(min_start, max_start + 1, (traj.shape[0],))
    return crop(traj, start_idx, crop_len)


@torch.no_grad()
def random_cond_contra_cropping(traj: torch.Tensor, crop_len: int, no_align: bool = True):
    t_len = traj.shape[1]
    assert t_len >= crop_len

    min_start = min(50, max(0, t_len - crop_len))
    max_start = t_len - crop_len
    if max_start <= min_start:
        start_idx_1 = torch.zeros((traj.shape[0],), dtype=torch.long)
    else:
        start_idx_1 = torch.randint(min_start, max_start + 1, (traj.shape[0],))

    crop_1 = crop(traj, start_idx_1, crop_len)

    if no_align and max_start > min_start:
        valid = np.arange(min_start, max_start + 1)
        avoid = np.arange(int(start_idx_1[0]), int(start_idx_1[0]) + crop_len)
        candidates = np.setdiff1d(valid, avoid)
        if candidates.size > 0:
            start2 = int(candidates[np.random.randint(0, candidates.size)])
            start_idx_2 = torch.tensor([start2], dtype=torch.long).expand(traj.shape[0])
        else:
            start_idx_2 = start_idx_1.clone()
    else:
        if max_start <= min_start:
            start_idx_2 = torch.zeros((traj.shape[0],), dtype=torch.long)
        else:
            start_idx_2 = torch.randint(min_start, max_start + 1, (traj.shape[0],))

    crop_2 = crop(traj, start_idx_2, crop_len)
    return crop_1[0], crop_2[0]


# -----------------------------------------------------------------------------
# Dataset classes
# -----------------------------------------------------------------------------

class L63TrainingData(Dataset):
    def __init__(
        self,
        crop_T: int,
        data_folder: str | None = None,
        train_size: int = 100,
        train_operator: bool = True,
        validation: bool = False,
        noisy_scale: float = 0.0,
        n_crops: int | None = None,
    ):
        self.crop_T = int(crop_T)
        self.train_operator = bool(train_operator)
        self.noisy_scale = float(noisy_scale)

        if data_folder is not None:
            self.data_path = data_folder
        elif validation:
            self.data_path = os.path.join(parent, 'l63_data_x', 'l63_data_val')
        else:
            self.data_path = os.path.join(parent, 'l63_data_x', 'l63_data_train')

        self.data_list = _resolve_l63_file_list(self.data_path, self.noisy_scale)
        self.params = torch.load(os.path.join(self.data_path, 'training_params.pth'), map_location='cpu', weights_only=False).float()

        self.data_list = self.data_list[:train_size]
        self.params = self.params[:train_size]

        sample = torch.load(self.data_list[0], map_location='cpu', weights_only=False)
        traj_sample = sample['1']
        self.T, self.D = int(traj_sample.shape[0]), int(traj_sample.shape[1])

        if n_crops is not None:
            self.n_crops = int(n_crops)
        else:
            crops_per_file = max(1, self.T // max(self.crop_T, 1))
            self.n_crops = len(self.data_list) * crops_per_file

        print(
            f'L63TrainingData | path={self.data_path} | n_files={len(self.data_list)} | '
            f'n_crops={self.n_crops} | T={self.T} | D={self.D} | noisy_scale={self.noisy_scale}'
        )

    def __len__(self):
        return self.n_crops

    def __getitem__(self, idx: int):
        file_idx = idx % len(self.data_list)
        sample = torch.load(self.data_list[file_idx], map_location='cpu', weights_only=False)
        params = sample['0'].float()               # (3,)
        traj = sample['1'].float()                 # (T, 3)

        if self.train_operator:
            if self.crop_T < self.T:
                cropped = random_cropping(traj[None], self.crop_T)[0]
            else:
                cropped = traj
            return params, cropped

        crop_1, crop_2 = random_cond_contra_cropping(traj[None], self.crop_T)
        return params, crop_1, crop_2


class L63TestingData(Dataset):
    def __init__(self, data_folder: str | None = None, noisy_scale: float = 0.0):
        if data_folder is not None:
            self.data_path = data_folder
        else:
            self.data_path = os.path.join(parent, 'l63_data_x', 'l63_data_test')

        self.noisy_scale = float(noisy_scale)
        self.data_list = _resolve_l63_file_list(self.data_path, self.noisy_scale)
        self.params = torch.load(os.path.join(self.data_path, 'training_params.pth'), map_location='cpu', weights_only=False).float()

        sample = torch.load(self.data_list[0], map_location='cpu', weights_only=False)
        traj_sample = sample['1']
        self.T, self.D = int(traj_sample.shape[0]), int(traj_sample.shape[1])

        print(
            f'L63TestingData | path={self.data_path} | n={len(self.data_list)} | '
            f'T={self.T} | D={self.D} | noisy_scale={self.noisy_scale}'
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int):
        sample = torch.load(self.data_list[idx], map_location='cpu', weights_only=False)
        params = sample['0'].float()
        traj = sample['1'].float()
        return params, traj


# -----------------------------------------------------------------------------
# Noise helpers
# -----------------------------------------------------------------------------

def _noise_suffix(noisy_scale: float) -> str:
    return f'_noise_{float(noisy_scale):.2f}.pth'


def _resolve_l63_file_list(data_path: str, noisy_scale: float):
    if noisy_scale > 0:
        pattern = os.path.join(data_path, f'0*{_noise_suffix(noisy_scale)}')
    else:
        pattern = os.path.join(data_path, '0*.pth')

    data_list = sorted(glob.glob(pattern))
    if noisy_scale > 0:
        data_list = [p for p in data_list if '_noise_' in os.path.basename(p)]
    else:
        data_list = [p for p in data_list if '_noise_' not in os.path.basename(p)]

    if not data_list:
        raise FileNotFoundError(
            f'No L63 data files found for noisy_scale={noisy_scale} in {data_path}. '
            f'Generate noise files with: '
            f'python dataloader/dataloader_l63.py --data_path {data_path} --noisy_scale {noisy_scale}'
        )
    return data_list


def save_noisy_l63_data(data_path: str, noisy_scale: float, overwrite: bool = False):
    clean_files = sorted(glob.glob(os.path.join(data_path, '0*.pth')))
    clean_files = [p for p in clean_files if '_noise_' not in os.path.basename(p)]
    if not clean_files:
        raise FileNotFoundError(f'No clean L63 files found in {data_path}')

    suffix = _noise_suffix(noisy_scale)
    written = 0
    skipped = 0

    for clean_path in clean_files:
        noisy_path = clean_path[:-4] + suffix
        if os.path.exists(noisy_path) and not overwrite:
            skipped += 1
            continue

        sample = torch.load(clean_path, map_location='cpu', weights_only=False)
        traj = sample['1'].float()

        # Match the KS/L96 convention: per-dimension std scaling.
        std_traj = traj.std(dim=0, keepdim=True)
        noise = noisy_scale * std_traj * torch.randn_like(traj)

        noisy_sample = {
            '0': sample['0'].float(),
            '1': traj + noise,
        }
        if 'ic' in sample:
            noisy_sample['ic'] = sample['ic'].float()

        torch.save(noisy_sample, noisy_path)
        written += 1

    print(
        f'save_noisy_l63_data | path={data_path} | noisy_scale={noisy_scale} | '
        f'written={written} | skipped={skipped}'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Precompute offline noisy L63 datasets')
    parser.add_argument('--data_path', action='append', required=True,
                        help='Dataset folder containing clean 000000.pth-style files.')
    parser.add_argument('--noisy_scale', type=float, required=True,
                        help='Noise scale to precompute, e.g. 0.3')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing noisy files if present.')
    cli_args = parser.parse_args()

    for path in cli_args.data_path:
        save_noisy_l63_data(path, cli_args.noisy_scale, overwrite=cli_args.overwrite)
