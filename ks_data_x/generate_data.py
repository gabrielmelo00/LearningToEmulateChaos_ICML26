"""
ks_data_x/generate_data.py
--------------------------
Generate train / validation / test trajectory datasets for the KS experiment.
Equivalent to l96_data_x/generate_data.py in the L96 pipeline.

Key differences from L96:
- No physical parameter F is sampled. The operator input is the IC u0.
- ICs are sampled as random low-wavenumber Fourier perturbations; the
  burn-in inside generate_ks_data ensures all trajectories land on the
  attractor before recording starts.
- State dimension is N=256 (spatial grid points).
- Trajectories are stored as (N,)-shaped snapshots, not (60,) L96 states.

Dataset layout (mirrors L96 structure):
    {data_folder}/{:06d}.pth   ->  {'0': ic (N,), '1': traj (T_stored, N)}
    {data_folder}/training_params.pth  ->  tensor of ICs, shape (n_samples, N)

'training_params' stores the ICs (operator inputs) so the dataloader can
access them without re-loading every trajectory file.
"""

import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from ks import generate_ks_data, KS

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='KS dataset generation')
parser.add_argument('--N',           default=256,  type=int,   help='Spatial grid points')
parser.add_argument('--L',           default=22,   type=float, help='Domain length parameter')
parser.add_argument('--dt',          default=0.25, type=float, help='Solver time step')
parser.add_argument('--T_burnin',    default=500., type=float, help='Burn-in time units')
parser.add_argument('--T_total',     default=500., type=float, help='Recording time units')
parser.add_argument('--t_res',       default=2,    type=int,   help='Subsampling factor')
parser.add_argument('--n_workers',   default=min(cpu_count(), 32), type=int)
args = parser.parse_args()

N      = args.N
L      = args.L
dt     = args.dt
T_burnin = args.T_burnin
T_total  = args.T_total
t_res    = args.t_res

T_stored = int(T_total / (dt * t_res))
print(f'State dim N={N} | T_stored={T_stored} per trajectory')


# ── IC sampling ───────────────────────────────────────────────────────────────

def sample_ic(seed, N=256, L=50, n_modes=4):
    """
    Sample a random low-wavenumber IC.
    The specific IC only affects the burn-in transient — after burn-in
    the solver is on the attractor regardless of this starting point.
    """
    np.random.seed(seed)
    x  = np.linspace(0, 2*np.pi*L, N, endpoint=False)
    ic = np.zeros(N)
    for k in range(1, n_modes + 1):
        amp = np.random.randn()
        phi = np.random.uniform(0, 2*np.pi)
        ic += amp * np.cos(k * x / L + phi)
    return (ic - ic.mean()).astype(np.float32)


def _worker(packed_args):
    """
    packed_args: np.ndarray of shape (N+1,)
        [ic (N,), seed (1,)]
    """
    return generate_ks_data(
        packed_args,
        L=L, N=N, dt=dt,
        T_burnin=T_burnin,
        T_total=T_total,
        t_res=t_res,
    )


# ── Dataset generation helper ─────────────────────────────────────────────────

def generate_split(data_folder, n_samples, seed_offset, chunk_size=50):
    """
    Generate n_samples trajectories and save to data_folder.

    Parameters
    ----------
    data_folder : str
    n_samples   : int
    seed_offset : int   — offset so train/val/test seeds never overlap
    chunk_size  : int   — trajectories generated per Pool call
    """
    os.makedirs(data_folder, exist_ok=True)

    ics = np.stack([
        sample_ic(seed_offset + i, N=N, L=L)
        for i in range(n_samples)
    ])                                              # (n_samples, N)
    torch.save(torch.from_numpy(ics), f'{data_folder}/training_params.pth')
    print(f'[{data_folder}] Saved {n_samples} ICs.')

    n_chunks = int(np.ceil(n_samples / chunk_size))
    for chunk_idx in tqdm(range(n_chunks), desc=data_folder):
        start = chunk_idx * chunk_size
        end   = min(start + chunk_size, n_samples)

        # Pack [ic, seed] for each sample in the chunk
        packed = np.concatenate([
            ics[start:end],                         # (chunk, N)
            (seed_offset + np.arange(start, end))[:, None].astype(np.float64)
        ], axis=1)                                  # (chunk, N+1)

        with Pool(args.n_workers) as pool:
            trajs = pool.map(_worker, packed)       # list of (T_stored, N)

        for local_ix, global_ix in enumerate(range(start, end)):
            torch.save(
                {'0': torch.from_numpy(ics[global_ix]),          # IC  (N,)
                 '1': torch.from_numpy(trajs[local_ix])},        # traj (T_stored, N)
                '{}/{:06d}.pth'.format(data_folder, global_ix)
            )

    print(f'[{data_folder}] Done. {n_samples} files written.')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    generate_split(
        data_folder='ks_data_train',
        n_samples=2000,
        seed_offset=0,
    )

    generate_split(
        data_folder='ks_data_val',
        n_samples=100,
        seed_offset=5000,
    )

    generate_split(``
        data_folder='ks_data_test',
        n_samples=200,
        seed_offset=10000,
    )