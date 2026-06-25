"""
eval_scripts/eval_ks.py
-----------------------
Evaluation loop for the KS emulator.
Equivalent to eval_scripts/eval_l96.py in the L96 pipeline.

Key differences from L96:
- State dimension is N (spatial grid) instead of 60.
- Statistics are (du/dt, du/dx, d²u/dx²) instead of L96 advection stats.
- Multi-step rollout is purely autoregressive — there is no parameter
  conditioning to worry about.

Metrics computed:
    1. One-step relative L2 error  (RMSE, normalised)
    2. Energy spectrum distance     (relative L1 on power spectra)
    3. Joint 3D distributional L1  (chi-score on (du/dt, du/dx, d²u/dx²))
"""

import numpy as np
import torch
import torch.utils.data
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

current = os.path.dirname(os.path.realpath(__file__))
parent  = os.path.dirname(current)
sys.path.append(parent)

from scripts.cal_stats_ks import cal_stats_l1_score_ks
from scripts.train_utils import LpLoss_


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def spectrum(u):
    """
    Power spectrum of a batch of spatial fields.

    Parameters
    ----------
    u : torch.Tensor, shape (T, N)

    Returns
    -------
    torch.Tensor, shape (N//2 + 1,) — time-averaged power spectrum
    """
    u_hat = torch.fft.rfft(u, dim=-1)
    return (u_hat.abs()**2).mean(dim=0)


def get_all_data(dataset, num_workers=4, shuffle=False):
    dataset_len = len(dataset)
    if dataset_len == 0:
        raise ValueError("Dataset is empty: len(dataset) == 0. Check your data folder and configuration.")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=dataset_len,
        num_workers=num_workers, shuffle=shuffle
    )
    for batch in loader:
        return batch


def save_ks_xt_heatmaps(true_traj, pred_traj, output_path, max_samples=3):
    """
    Save side-by-side x-t heatmaps for numerical vs emulator KS trajectories.

    Parameters
    ----------
    true_traj : torch.Tensor, shape (B, T, N)
    pred_traj : torch.Tensor, shape (B, T, N)
    output_path : str
        Directory where figures are written.
    max_samples : int
        Maximum number of trajectories to plot.
    """
    n_plot = min(max_samples, true_traj.shape[0])
    plot_dir = output_path if output_path else '.'
    os.makedirs(plot_dir, exist_ok=True)

    for i in range(n_plot):
        u_true = true_traj[i].detach().cpu().numpy()  # (T, N)
        u_pred = pred_traj[i].detach().cpu().numpy()  # (T, N)

        # Keep a shared color scale for fair visual comparison.
        vmin = min(u_true.min(), u_pred.min())
        vmax = max(u_true.max(), u_pred.max())

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

        im0 = axes[0].imshow(u_true, aspect='auto', origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[0].set_title('Numerical (Ground Truth)')
        axes[0].set_xlabel('x index')
        axes[0].set_ylabel('t index')

        im1 = axes[1].imshow(u_pred, aspect='auto', origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1].set_title('Emulator (Predicted)')
        axes[1].set_xlabel('x index')
        axes[1].set_ylabel('t index')

        fig.colorbar(im1, ax=axes, shrink=0.9, label='u(x,t)')

        save_path = os.path.join(plot_dir, f'ks_xt_compare_sample_{i:03d}.png')
        fig.savefig(save_path, dpi=180)
        plt.close(fig)


# ─────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────

def eval_ks(operator, args, noisy_scale=0, x_len=100, output_path=''):
    """
    Evaluate a KS emulator following the same noisy-train/clean-test
    protocol as the L96 pipeline.

    Parameters
    ----------
    operator : nn.Module
        The trained emulator. Expected signature:
            u_next = operator(u_now)
        where u_now has shape (B, 1, N) and u_next has shape (B, 1, N).
        (Matches the FNO convention used for L96.)
    args : argparse.Namespace
        Must contain: args.gpu (int).
    x_len : int
        Number of autoregressive rollout steps for trajectory evaluation.
    output_path : str
        Directory where result .txt files are written.
    """
    from dataloader.dataloader_ks import KSTestingData, KSTrainingData

    def noise_tag(scale):
        return f'ns{float(scale):.2f}'

    noisy_eval_split = getattr(args, 'ks_noisy_eval_split', 'test')

    if noisy_scale == 0:
        eval_size = 200
        initial_dataset = KSTestingData(
            data_folder=getattr(args, 'ks_data_test', None),
            noisy_scale=getattr(args, 'ks_clean_eval_init_noise', 0.0),
        )
        eval_dataset = KSTestingData(
            data_folder=getattr(args, 'ks_data_test', None),
            noisy_scale=0.0,
        )
        eval_name = 'test_on_clean_data'
        result_filename = 'Results_test_on_clean_data.txt'
    else:
        if noisy_eval_split == 'validation':
            eval_size = 100
            initial_dataset = KSTrainingData(
                crop_T=5000,
                data_folder=getattr(args, 'ks_data_val', None),
                train_size=eval_size,
                train_operator=True,
                validation=True,
                noisy_scale=noisy_scale,
            )
            eval_dataset = KSTrainingData(
                crop_T=5000,
                data_folder=getattr(args, 'ks_data_val', None),
                train_size=eval_size,
                train_operator=True,
                validation=True,
                noisy_scale=noisy_scale,
            )
            eval_name = f'validation_on_noise_data_{noise_tag(noisy_scale)}'
            result_filename = 'Results_validation_on_noise_data.txt'
        else:
            eval_size = 200
            initial_dataset = KSTestingData(
                data_folder=getattr(args, 'ks_data_test', None),
                noisy_scale=noisy_scale,
            )
            eval_dataset = KSTestingData(
                data_folder=getattr(args, 'ks_data_test', None),
                noisy_scale=noisy_scale,
            )
            eval_name = f'test_on_noise_data_{noise_tag(noisy_scale)}'
            result_filename = 'Results_test_on_noise_data.txt'

    all_data_initial = get_all_data(initial_dataset)
    all_data_eval = get_all_data(eval_dataset)
    ics_initial, trajs_initial = all_data_initial[0], all_data_initial[1]
    ics_eval, trajs_eval = all_data_eval[0], all_data_eval[1]
    eval_size = min(eval_size, len(trajs_eval))
    ics_initial = ics_initial[:eval_size]
    trajs_initial = trajs_initial[:eval_size]
    ics_eval = ics_eval[:eval_size]
    trajs_eval = trajs_eval[:eval_size]

    N = ics_eval.shape[1]
    T = trajs_eval.shape[1]

    eval_root = output_path if output_path else '.'
    rollout_dir = f'rollout_{x_len}'
    eval_dir = os.path.join(eval_root, 'eval_noisy_trainval_clean_test', rollout_dir, eval_name)
    os.makedirs(eval_dir, exist_ok=True)

    dist_spectrum  = []
    slow_stats_list = []

    # ── Rollout for attractor metrics ─────────────────────────────────────────
    # Mirrors eval_l96.py: anchored 2-step rollout for x_len==100 (training-
    # length diagnostics), free-running for longer rollouts (attractor fidelity).
    x_0_start = 50                                 # skip very start of trajectory
    x_end     = x_0_start + x_len

    with torch.no_grad():
        operator.eval()

        u0 = trajs_initial[:, x_0_start, :][:, None, :].to(args.gpu)  # (B, 1, N)
        rollout = [u0]

        if x_len == 100:
            # Anchored: even steps reset from GT, odd steps are operator predictions
            for t in range(1, x_len):
                if t % 2 == 0:
                    u_next = trajs_initial[:, x_0_start + t, :][:, None, :].to(args.gpu)
                else:
                    u_next = operator(u0)
                rollout.append(u_next)
                u0 = u_next
        else:
            # Free-running: purely autoregressive — measures attractor fidelity
            for t in range(1, x_len):
                u_next = operator(u0)
                rollout.append(u_next)
                u0 = u_next

        out_traj  = torch.cat(rollout, dim=1)      # (B, x_len, N)
        true_traj = trajs_eval[:, x_0_start:x_end, :].to(args.gpu)  # (B, ?, N)

        # Truncate to the shorter of the two so the histogram metric compares
        # equal-length trajectories.  true_traj can be shorter than x_len when
        # the stored trajectory (T) is shorter than x_0_start + x_len.
        compare_len = min(out_traj.shape[1], true_traj.shape[1])
        if compare_len < x_len:
            print(f'[eval_ks] WARNING: true trajectory shorter than rollout '
                  f'(T={T}, x_0_start={x_0_start}, x_len={x_len}). '
                  f'Metrics computed on {compare_len} steps.')
        out_traj  = out_traj [:, :compare_len, :]
        true_traj = true_traj[:, :compare_len, :]

        # Save qualitative x-t heatmaps for numerical vs emulator trajectories.
        # Truncate to training rollout length so both panels share the same t-axis scale.
        n_plot = int(getattr(args, 'ks_plot_samples', 3))
        plot_t = min(getattr(args, 'x_len', x_len), x_len)
        save_ks_xt_heatmaps(true_traj[:, :plot_t], out_traj[:, :plot_t], output_path=eval_dir, max_samples=n_plot)

        for i in range(eval_size):
            u_true = true_traj[i]                  # (x_len, N)
            u_pred = out_traj[i]                   # (x_len, N)

            # Energy spectrum distance
            spec_true = spectrum(u_true)
            spec_pred = spectrum(u_pred)
            rel_diff  = (spec_true - spec_pred).abs() / spec_true.sum()
            dist_spectrum.append(rel_diff.sum().cpu().item())

            # Joint 3D distributional L1 score
            l1 = cal_stats_l1_score_ks(
                u_true.cpu().numpy(),
                u_pred.cpu().numpy()
            )
            slow_stats_list.append(l1)

    spectrum_stats = {
        'q25': np.quantile(dist_spectrum, 0.25),
        'q50': np.quantile(dist_spectrum, 0.50),
        'q75': np.quantile(dist_spectrum, 0.75),
    }
    l1_stats = {
        'q25': np.quantile(slow_stats_list, 0.25),
        'q50': np.quantile(slow_stats_list, 0.50),
        'q75': np.quantile(slow_stats_list, 0.75),
    }
    print(f'Spectrum distance (q25/q50/q75): '
          f'{spectrum_stats["q25"]:.4f} / {spectrum_stats["q50"]:.4f} / {spectrum_stats["q75"]:.4f}')
    print(f'L1 score (q25/q50/q75): '
          f'{l1_stats["q25"]:.4f} / {l1_stats["q50"]:.4f} / {l1_stats["q75"]:.4f}')

    # ── One-step RMSE ─────────────────────────────────────────────────────────
    eval_bsize = 5
    chunk_size = 100  # process time steps in chunks to avoid OOM
    l2_lp_list = []

    with torch.no_grad():
        operator.eval()
        for s in range(0, eval_size, eval_bsize):
            e = min(s + eval_bsize, eval_size)

            traj_initial_b = trajs_initial[s:e]
            traj_eval_b = trajs_eval[s:e]
            
            # Process in chunks to avoid OOM
            for t_start in range(0, T - 1, chunk_size):
                t_end = min(t_start + chunk_size, T - 1)
                u_now  = traj_initial_b[:, t_start:t_end].reshape(-1, 1, N).to(args.gpu)
                u_next_true = traj_eval_b[:, t_start+1:t_end+1].reshape(-1, N).to(args.gpu)

                u_next_pred = operator(u_now).squeeze(1)         # (chunk, N)

                l2_lp_list.append(
                    LpLoss_(2).rel(u_next_pred, u_next_true).cpu().item()
                )
                
                del u_now, u_next_true, u_next_pred
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if len(l2_lp_list) == 0:
        raise ValueError(f'No one-step KS evaluation samples were processed (eval_size={eval_size}).')
    rmse = np.mean(l2_lp_list)
    print(f'One-step relative L2 (RMSE): {rmse:.6f}')

    l2_lp_array = np.asarray(l2_lp_list, dtype=float)

    # ── Write results ─────────────────────────────────────────────────────────
    result_path = os.path.join(eval_dir, result_filename)
    with open(result_path, 'w') as f:
        f.write(f'noise {noisy_scale} with eval length {x_len} training length {args.x_len} \n')
        f.write(f'mse_[\'rMSE\', {np.quantile(l2_lp_array, 0.25)}, '
                f'\'50 percentile\', {np.quantile(l2_lp_array, 0.5)}, '
                f'{np.quantile(l2_lp_array, 0.75)}] \n ')
        f.write(f'l1_3d_score: {[np.mean(slow_stats_list), np.quantile(slow_stats_list, 0.25), "50 percentile", np.quantile(slow_stats_list, 0.5), np.quantile(slow_stats_list, 0.75)]} \n ')
        f.write(f'spectrum distance:{[np.quantile(np.array(dist_spectrum), 0.25), np.array([50]).reshape(-1,1), np.quantile(np.array(dist_spectrum), 0.5), np.quantile(np.array(dist_spectrum), 0.75)]} \n \n')

    if noisy_scale == 0:
        rollout_result_path = os.path.join(eval_root, f'Results_ks_test_rollout_{x_len}.txt')
        with open(rollout_result_path, 'w') as f:
            f.write(f'KS evaluation | rollout_len={x_len} | N={N}\n\n')
            f.write(f'One-step relative L2 (RMSE): {rmse:.6f}\n')
            f.write(f'Spectrum distance  q25/q50/q75: '
                    f'{spectrum_stats["q25"]:.4f} / {spectrum_stats["q50"]:.4f} / {spectrum_stats["q75"]:.4f}\n')
            f.write(f'L1 score           q25/q50/q75: '
                    f'{l1_stats["q25"]:.4f} / {l1_stats["q50"]:.4f} / {l1_stats["q75"]:.4f}\n')
        eval_lengths = [int(x) for x in getattr(args, 'ks_eval_lengths', [x_len])]
        if x_len == max(eval_lengths):
            legacy_result_path = os.path.join(eval_root, 'Results_ks_test.txt')
            with open(legacy_result_path, 'w') as f:
                f.write(f'KS evaluation | rollout_len={x_len} | N={N}\n\n')
                f.write(f'One-step relative L2 (RMSE): {rmse:.6f}\n')
                f.write(f'Spectrum distance  q25/q50/q75: '
                        f'{spectrum_stats["q25"]:.4f} / {spectrum_stats["q50"]:.4f} / {spectrum_stats["q75"]:.4f}\n')
                f.write(f'L1 score           q25/q50/q75: '
                        f'{l1_stats["q25"]:.4f} / {l1_stats["q50"]:.4f} / {l1_stats["q75"]:.4f}\n')

    return rmse, spectrum_stats, l1_stats
