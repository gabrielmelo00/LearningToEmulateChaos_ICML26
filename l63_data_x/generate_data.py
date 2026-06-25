"""
Generate Lorenz-63 train/val/test datasets.

Output layout mirrors existing pipelines:
    {split_dir}/{:06d}.pth  -> {'0': params(3,), '1': traj(T,3), 'ic': ic(3,)}
    {split_dir}/training_params.pth -> tensor (n_samples, 3)

The default setup follows the requested benchmark:
- dt = 0.01
- n_steps = 10_000 per crop
- stride = 5 (stored length 2,000)
- training split size = 100
- additive noisy copies are produced separately by dataloader/dataloader_l63.py
"""

from __future__ import annotations

import argparse
import os
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
from tqdm import tqdm

from l63 import (
    DEFAULT_BETA,
    DEFAULT_RHO,
    DEFAULT_SIGMA,
    generate_l63_segment,
    sample_l63_ic,
    sample_l63_params,
)


def _worker(payload: tuple[np.ndarray, np.ndarray, float, int, int, int]) -> np.ndarray:
    params, ic, dt, n_steps, stride, burnin_steps = payload
    traj, _ = generate_l63_segment(
        params=params,
        initial_state=ic,
        dt=dt,
        n_steps=n_steps,
        stride=stride,
        burnin_steps=burnin_steps,
    )
    return traj


def _build_samples(
    n_samples: int,
    seed_offset: int,
    sigma_center: float,
    rho_center: float,
    beta_center: float,
    sigma_delta: float,
    rho_delta: float,
    beta_delta: float,
    ic_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    params_list = []
    ic_list = []
    for i in range(n_samples):
        seed = seed_offset + i
        params = sample_l63_params(
            seed=seed,
            sigma_center=sigma_center,
            rho_center=rho_center,
            beta_center=beta_center,
            sigma_delta=sigma_delta,
            rho_delta=rho_delta,
            beta_delta=beta_delta,
        )
        ic = sample_l63_ic(seed=seed + 1_000_000, scale=ic_scale)
        params_list.append(params)
        ic_list.append(ic)
    return np.asarray(params_list, dtype=np.float32), np.asarray(ic_list, dtype=np.float32)


def generate_split(
    data_folder: str,
    n_samples: int,
    seed_offset: int,
    dt: float,
    n_steps: int,
    stride: int,
    burnin_steps: int,
    sigma_center: float,
    rho_center: float,
    beta_center: float,
    sigma_delta: float,
    rho_delta: float,
    beta_delta: float,
    ic_scale: float,
    n_workers: int,
    chunk_size: int,
) -> None:
    os.makedirs(data_folder, exist_ok=True)

    params, initial_conditions = _build_samples(
        n_samples=n_samples,
        seed_offset=seed_offset,
        sigma_center=sigma_center,
        rho_center=rho_center,
        beta_center=beta_center,
        sigma_delta=sigma_delta,
        rho_delta=rho_delta,
        beta_delta=beta_delta,
        ic_scale=ic_scale,
    )

    torch.save(torch.from_numpy(params), os.path.join(data_folder, 'training_params.pth'))
    torch.save(torch.from_numpy(initial_conditions), os.path.join(data_folder, 'initial_conditions.pth'))

    payloads = [
        (params[i], initial_conditions[i], dt, n_steps, stride, burnin_steps)
        for i in range(n_samples)
    ]

    n_chunks = int(np.ceil(n_samples / chunk_size))
    for chunk_idx in tqdm(range(n_chunks), desc=f'generating {os.path.basename(data_folder)}'):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_samples)
        chunk_payloads = payloads[start:end]

        if n_workers > 1:
            with Pool(processes=n_workers) as pool:
                trajs = pool.map(_worker, chunk_payloads)
        else:
            trajs = list(map(_worker, chunk_payloads))

        for local_ix, global_ix in enumerate(range(start, end)):
            torch.save(
                {
                    '0': torch.from_numpy(params[global_ix]),
                    '1': torch.from_numpy(trajs[local_ix]),
                    'ic': torch.from_numpy(initial_conditions[global_ix]),
                },
                os.path.join(data_folder, f'{global_ix:06d}.pth'),
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lorenz-63 dataset generation')
    parser.add_argument('--output_root', default='l63_data_x', type=str)
    parser.add_argument('--train_folder', default='l63_data_train', type=str)
    parser.add_argument('--val_folder', default='l63_data_val', type=str)
    parser.add_argument('--test_folder', default='l63_data_test', type=str)

    parser.add_argument('--n_train', default=100, type=int)
    parser.add_argument('--n_val', default=20, type=int)
    parser.add_argument('--n_test', default=20, type=int)

    parser.add_argument('--dt', default=0.01, type=float)
    parser.add_argument('--n_steps', default=10_000, type=int)
    parser.add_argument('--stride', default=5, type=int)
    parser.add_argument('--burnin_steps', default=5_000, type=int)

    parser.add_argument('--sigma_center', default=DEFAULT_SIGMA, type=float)
    parser.add_argument('--rho_center', default=DEFAULT_RHO, type=float)
    parser.add_argument('--beta_center', default=DEFAULT_BETA, type=float)
    parser.add_argument('--sigma_delta', default=1.0, type=float)
    parser.add_argument('--rho_delta', default=2.0, type=float)
    parser.add_argument('--beta_delta', default=0.2, type=float)
    parser.add_argument('--ic_scale', default=8.0, type=float)

    parser.add_argument('--n_workers', default=min(cpu_count(), 16), type=int)
    parser.add_argument('--chunk_size', default=20, type=int)
    args = parser.parse_args()

    root = args.output_root
    os.makedirs(root, exist_ok=True)

    print(
        f'Generating L63 dataset | dt={args.dt} | steps={args.n_steps} | '
        f'stride={args.stride} | stored_len={args.n_steps // args.stride}'
    )

    generate_split(
        data_folder=os.path.join(root, args.train_folder),
        n_samples=args.n_train,
        seed_offset=0,
        dt=args.dt,
        n_steps=args.n_steps,
        stride=args.stride,
        burnin_steps=args.burnin_steps,
        sigma_center=args.sigma_center,
        rho_center=args.rho_center,
        beta_center=args.beta_center,
        sigma_delta=args.sigma_delta,
        rho_delta=args.rho_delta,
        beta_delta=args.beta_delta,
        ic_scale=args.ic_scale,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
    )

    generate_split(
        data_folder=os.path.join(root, args.val_folder),
        n_samples=args.n_val,
        seed_offset=50_000,
        dt=args.dt,
        n_steps=args.n_steps,
        stride=args.stride,
        burnin_steps=args.burnin_steps,
        sigma_center=args.sigma_center,
        rho_center=args.rho_center,
        beta_center=args.beta_center,
        sigma_delta=args.sigma_delta,
        rho_delta=args.rho_delta,
        beta_delta=args.beta_delta,
        ic_scale=args.ic_scale,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
    )

    generate_split(
        data_folder=os.path.join(root, args.test_folder),
        n_samples=args.n_test,
        seed_offset=100_000,
        dt=args.dt,
        n_steps=args.n_steps,
        stride=args.stride,
        burnin_steps=args.burnin_steps,
        sigma_center=args.sigma_center,
        rho_center=args.rho_center,
        beta_center=args.beta_center,
        sigma_delta=args.sigma_delta,
        rho_delta=args.rho_delta,
        beta_delta=args.beta_delta,
        ic_scale=args.ic_scale,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
    )

    print('Done.')
