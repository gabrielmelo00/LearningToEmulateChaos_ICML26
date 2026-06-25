import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from ks import generate_ks_segment, sample_ic


parser = argparse.ArgumentParser(description='Single-trajectory KS data generation')
parser.add_argument('--N', default=256, type=int, help='Spatial dimension')
parser.add_argument('--L', default=50, type=float, help='Domain length parameter')
parser.add_argument('--dt', default=0.1, type=float, help='Solver timestep')
parser.add_argument('--t_res', default=10, type=int, help='Save every t_res integration steps')
parser.add_argument('--T_segment', default=2000.0, type=float, help='Simulation time per segment')
parser.add_argument('--n_burnin', default=20, type=int, help='Number of burn-in segments before saving')
parser.add_argument('--seed', default=0, type=int, help='Random seed for initial condition')
parser.add_argument('--n_train', default=10, type=int, help='Number of train segments')
parser.add_argument('--n_val', default=1, type=int, help='Number of validation segments')
parser.add_argument('--n_test', default=2, type=int, help='Number of test segments')
parser.add_argument('--train_folder', default='ks_single_traj_train', type=str)
parser.add_argument('--val_folder', default='ks_single_traj_val', type=str)
parser.add_argument('--test_folder', default='ks_single_traj_test', type=str)
args = parser.parse_args()


def integrate_segment(ic, L, N, T, dt, t_res):
    """Integrate one contiguous segment. Returns (saved_traj, final_state)."""
    return generate_ks_segment(
        initial_condition=ic,
        L=L,
        N=N,
        dt=dt,
        T=T,
        t_res=t_res,
    )


ic = sample_ic(seed=args.seed, N=args.N, L=args.L)

print(f'Running {args.n_burnin} burn-in segments...')
for _ in range(args.n_burnin):
    _, ic = integrate_segment(ic, args.L, args.N, args.T_segment, args.dt, args.t_res)

splits = [
    (args.train_folder, args.n_train),
    (args.val_folder, args.n_val),
    (args.test_folder, args.n_test),
]

for data_folder, n_samples in splits:
    os.makedirs(data_folder, exist_ok=True)

    existing = sorted(
        [f for f in os.listdir(data_folder) if f.endswith('.pth') and f != 'training_params.pth']
    )
    n_existing = len(existing)
    n_missing = n_samples - n_existing
    print(f'\n[{data_folder}] {n_existing}/{n_samples} files already exist, {n_missing} to generate.')

    segment_ics = []
    for ix in tqdm(range(n_samples), desc=data_folder):
        segment_ic = ic.copy()
        traj, ic = integrate_segment(segment_ic, args.L, args.N, args.T_segment, args.dt, args.t_res)
        segment_ics.append(segment_ic.astype(np.float32))

        out_path = f'{data_folder}/{ix:06d}.pth'
        if not os.path.exists(out_path):
            torch.save(
                {
                    '0': torch.from_numpy(segment_ic.astype(np.float32)),
                    '1': torch.from_numpy(traj.astype(np.float32)),
                },
                out_path,
            )

    training_params = torch.from_numpy(np.stack(segment_ics, axis=0))
    torch.save(training_params, f'{data_folder}/training_params.pth')

    saved = len([f for f in os.listdir(data_folder) if f.endswith('.pth') and f != 'training_params.pth'])
    print(f'  -> Done. {saved}/{n_samples} files now present.')
