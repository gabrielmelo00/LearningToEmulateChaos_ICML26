"""
dataloader/dataloader_ks.py
---------------------------
PyTorch Dataset classes for the KS experiment.
Equivalent to dataloader/dataloader_l96.py in the L96 pipeline.

Key differences from L96:
- No physical parameter. __getitem__ returns (ic, traj) where ic is the
  initial condition (operator input, shape N) and traj is the trajectory
  (shape crop_T x N).
- State dimension is N=256 (spatial grid) instead of 60.
- noisy_scale selects pre-generated noisy files saved alongside the clean
  `.pth` trajectories, matching the L96 offline-noise workflow.
- 'training_params' stores ICs (shape n_samples x N), not scalar F values.

Cropping conventions mirror L96:
- random_cropping:            single random window from the trajectory.
- random_cond_contra_cropping: two non-overlapping windows (for contrastive).
"""

import argparse
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

current = os.path.dirname(os.path.realpath(__file__))
parent  = os.path.dirname(current)


# ─────────────────────────────────────────────
# Crop utilities  (identical logic to L96)
# ─────────────────────────────────────────────

def crop(traj, Tsidx, Tlen):
    """Extract a window of length Tlen starting at Tsidx[i] for each sample i."""
    assert Tsidx.shape[0] == traj.shape[0]
    return torch.stack([traj[i, Tsidx[i]: Tsidx[i] + Tlen] for i in range(Tsidx.shape[0])])


@torch.no_grad()
def random_cropping(traj, crop_T):
    """Return one random contiguous window of length crop_T per trajectory."""
    T = traj.shape[1]
    assert T >= crop_T
    Tsidx = torch.randint(50, T - crop_T, (traj.shape[0],))
    return crop(traj, Tsidx, crop_T)


@torch.no_grad()
def random_cond_contra_cropping(traj, crop_T, no_align=True):
    """
    Return two non-overlapping windows of length crop_T from the same trajectory.
    Used for contrastive learning objectives.
    """
    T = traj.shape[1]
    assert T >= crop_T

    Tsidx_1 = torch.randint(50, T - crop_T, (traj.shape[0],))
    crop_1   = crop(traj, Tsidx_1, crop_T)

    if no_align:
        index    = np.arange(50, T - crop_T)
        in_index = np.arange(int(Tsidx_1[0]), int(Tsidx_1[0]) + crop_T)
        out_index = np.setdiff1d(index, in_index)
        Tsidx_2  = torch.tensor(
            [out_index[torch.randint(out_index.shape[0], (1,)).item()]]
        ).expand(traj.shape[0])
    else:
        Tsidx_2 = torch.randint(50, T - crop_T, (traj.shape[0],))

    crop_2 = crop(traj, Tsidx_2, crop_T)
    return crop_1[0], crop_2[0]


# ─────────────────────────────────────────────
# Dataset classes
# ─────────────────────────────────────────────

class KSTrainingData(Dataset):
    """
    Training / validation dataset for KS.

    Each sample file contains:
        {'0': ic   (torch.Tensor, shape N)        — operator input (IC)
         '1': traj (torch.Tensor, shape T x N)}   — trajectory on attractor

    Parameters
    ----------
    crop_T : int
        Length of trajectory window returned per sample.
    data_folder : str or None
        Path to the dataset folder. If None, uses default relative path.
    train_size : int
        Maximum number of samples to use.
    train_operator : bool
        If True, returns (ic, cropped_traj) for operator learning.
        If False, returns (ic, crop_1, crop_2) for contrastive learning.
    validation : bool
        If True, loads from the validation folder instead.
    noisy_scale : float
        Scale of Gaussian noise to add to trajectories (relative to std).
        0 means no noise (default).
    """

    def __init__(self,
                 crop_T,
                 data_folder=None,
                 train_size=1000,
                 train_operator=True,
                 validation=False,
                 noisy_scale=0.0,
                 n_crops=None):
        self.crop_T         = crop_T
        self.train_operator = train_operator
        self.noisy_scale    = noisy_scale

        if data_folder is not None:
            self.data_path = data_folder
        elif validation:
            self.data_path = os.path.join(parent, 'ks_data_x', 'ks_data_val')
        else:
            self.data_path = os.path.join(parent, 'ks_data_x', 'ks_data_train')

        self.data_list = _resolve_ks_file_list(self.data_path, self.noisy_scale)
        self.ics       = torch.load(os.path.join(self.data_path, 'training_params.pth'), weights_only=False)

        self.data_list = self.data_list[:train_size]
        self.ics       = self.ics[:train_size]

        sample      = torch.load(self.data_list[0], weights_only=False)
        traj_sample = sample['1']
        self.T, self.N = traj_sample.shape[0], traj_sample.shape[1]

        # n_crops: virtual dataset length = batch_size × batches_per_epoch,
        # computed externally in dataloader_init.py. When called directly
        # without this argument, fall back to the number of full crops per
        # file, but keep at least one sample per file so evaluation can use
        # full trajectories by passing crop_T >= T.
        if n_crops is not None:
            self.n_crops = n_crops
        else:
            crops_per_file = max(1, self.T // self.crop_T)
            self.n_crops = len(self.data_list) * crops_per_file

        print(f'KSTrainingData | path={self.data_path} | '
              f'n_files={len(self.data_list)} | n_crops={self.n_crops} | '
              f'T={self.T} | N={self.N}')

    def __len__(self):
        return self.n_crops

    def __getitem__(self, idx):
        # When n_crops > len(data_list), cycle over segment files so every
        # index maps to a valid file while still drawing a fresh random crop.
        file_idx = idx % len(self.data_list)
        d   = torch.load(self.data_list[file_idx], weights_only=False)
        ic  = d['0'].float()               # (N,)  — operator input
        traj = d['1'].float()              # (T, N)

        if self.train_operator:
            if self.crop_T < self.T:
                cropped = random_cropping(traj[None], self.crop_T)[0]  # (crop_T, N)
            else:
                cropped = traj                                          # (T, N)
            return ic, cropped
        else:
            crop_1, crop_2 = random_cond_contra_cropping(traj[None], self.crop_T)
            return ic, crop_1, crop_2


class KSTestingData(Dataset):
    """
    Test dataset for KS.

    Returns (ic, traj) without any cropping, so the full trajectory is
    available for multi-step rollout evaluation.

    Parameters
    ----------
    data_folder : str or None
        Path to test dataset folder. If None, uses default relative path.
    """

    def __init__(self, data_folder=None, noisy_scale=0.0):
        if data_folder is not None:
            self.data_path = data_folder
        else:
            self.data_path = os.path.join(parent, 'ks_data_x', 'ks_data_test')
        self.noisy_scale = noisy_scale

        self.data_list = _resolve_ks_file_list(self.data_path, self.noisy_scale)
        self.ics       = torch.load(os.path.join(self.data_path, 'training_params.pth'), weights_only=False)

        sample      = torch.load(self.data_list[0], weights_only=False)
        traj_sample = sample['1']
        self.T, self.N = traj_sample.shape[0], traj_sample.shape[1]

        print(f'KSTestingData | path={self.data_path} | '
              f'n={len(self.data_list)} | T={self.T} | N={self.N}')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        d    = torch.load(self.data_list[idx], weights_only=False)
        ic   = d['0'].float()              # (N,)
        traj = d['1'].float()              # (T, N)
        return ic, traj


def _noise_suffix(noisy_scale):
    return f'_noise_{noisy_scale:.2f}.pth'


def _resolve_ks_file_list(data_path, noisy_scale):
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
            f'No KS data files found for noisy_scale={noisy_scale} in {data_path}. '
            f'Generate them first with: '
            f'python dataloader/dataloader_ks.py --data_path {data_path} --noisy_scale {noisy_scale}'
        )
    return data_list


def save_noisy_ks_data(data_path, noisy_scale, overwrite=False):
    clean_files = sorted(glob.glob(os.path.join(data_path, '0*.pth')))
    clean_files = [p for p in clean_files if '_noise_' not in os.path.basename(p)]
    if not clean_files:
        raise FileNotFoundError(f'No clean KS files found in {data_path}')

    suffix = _noise_suffix(noisy_scale)
    written = 0
    skipped = 0
    for clean_path in clean_files:
        noisy_path = clean_path[:-4] + suffix
        if os.path.exists(noisy_path) and not overwrite:
            skipped += 1
            continue
        sample = torch.load(clean_path, weights_only=False)
        traj = sample['1'].float()
        std_traj = traj.std(dim=0, keepdim=True)
        noise = noisy_scale * std_traj * torch.randn_like(traj)
        noisy_sample = {
            '0': sample['0'].float(),
            '1': traj + noise,
        }
        torch.save(noisy_sample, noisy_path)
        written += 1
    print(
        f'save_noisy_ks_data | path={data_path} | noisy_scale={noisy_scale} | '
        f'written={written} | skipped={skipped}'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Precompute offline noisy KS datasets')
    parser.add_argument('--data_path', action='append', required=True,
                        help='Dataset folder containing clean 000000.pth-style KS files. '
                             'Pass multiple times for train/val/test.')
    parser.add_argument('--noisy_scale', type=float, required=True,
                        help='Noise scale to precompute, e.g. 0.3')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing noisy files if present.')
    cli_args = parser.parse_args()

    for path in cli_args.data_path:
        save_noisy_ks_data(path, cli_args.noisy_scale, overwrite=cli_args.overwrite)
