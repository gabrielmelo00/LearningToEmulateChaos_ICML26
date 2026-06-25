"""
Evaluation loop for Lorenz-63 emulator.

Outputs:
- noisy and clean evaluation result files (same folder pattern as KS),
- long-rollout attractor visualizations comparing emulator vs ground truth.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from eval_scripts.plot_l63 import (
    plot_l63_attractor_geometry_comparison,
)
from scripts.cal_stats_l63 import cal_l63_attractor_metrics, cal_stats_l1_score_l63
from scripts.train_utils import LpLoss_


def _noise_tag(scale: float) -> str:
    return f'ns{float(scale):.2f}'


def _resolve_l63_approach_label(args) -> str:
    loss_mode = str(getattr(args, 'loss_mode', '')).strip().lower()
    with_geomloss = int(getattr(args, 'with_geomloss', 0))
    prefix = str(getattr(args, 'prefix', '')).strip().lower()

    if loss_mode == 'learnable_ot' or prefix.startswith('wgan_'):
        return 'WGAN'
    if loss_mode == 'learnable_sinkhorn' or prefix.startswith('sinkhorn_'):
        return 'Learnable Sinkhorn'
    if (loss_mode == 'ot' and with_geomloss > 0) or prefix.startswith('ot_fixed_'):
        return 'Fixed OT'
    return 'MSE'


def _get_all_data(dataset, num_workers: int = 0, shuffle: bool = False):
    n = len(dataset)
    if n == 0:
        raise ValueError('Dataset is empty')
    loader = torch.utils.data.DataLoader(dataset, batch_size=n, num_workers=num_workers, shuffle=shuffle)
    for batch in loader:
        return batch
    raise RuntimeError('Failed to load evaluation batch')


def _save_l63_rollout_plots(
    true_traj: torch.Tensor,
    pred_traj: torch.Tensor,
    output_path: str,
    max_samples: int = 3,
    approach_label: str | None = None,
):
    os.makedirs(output_path, exist_ok=True)
    n_plot = min(max_samples, true_traj.shape[0])
    export_dpi = 1200

    for i in range(n_plot):
        true_np = true_traj[i].detach().cpu().numpy()
        pred_np = pred_traj[i].detach().cpu().numpy()

        fig = plot_l63_attractor_geometry_comparison(
            true_np,
            pred_np,
            approach_label=approach_label,
            show_titles=False,
        )
        fig.savefig(
            os.path.join(output_path, f'l63_attractor_geometry_{i:03d}.pdf'),
            dpi=export_dpi,
            bbox_inches='tight',
        )
        plt.close(fig)


def eval_l63(operator, args, noisy_scale: float = 0.0, x_len: int = 1000, output_path: str = ''):
    from dataloader.dataloader_l63 import L63TestingData, L63TrainingData

    noisy_eval_split = getattr(args, 'l63_noisy_eval_split', 'test')

    if noisy_scale == 0:
        eval_size = int(getattr(args, 'l63_eval_size_test', 20))
        initial_dataset = L63TestingData(
            data_folder=getattr(args, 'l63_data_test', None),
            noisy_scale=getattr(args, 'l63_clean_eval_init_noise', 0.0),
        )
        eval_dataset = L63TestingData(
            data_folder=getattr(args, 'l63_data_test', None),
            noisy_scale=0.0,
        )
        eval_name = 'test_on_clean_data'
        result_filename = 'Results_test_on_clean_data.txt'
    else:
        if noisy_eval_split == 'validation':
            eval_size = int(getattr(args, 'l63_eval_size_val', 20))
            initial_dataset = L63TrainingData(
                crop_T=10_000,
                data_folder=getattr(args, 'l63_data_val', None),
                train_size=eval_size,
                train_operator=True,
                validation=True,
                noisy_scale=noisy_scale,
            )
            eval_dataset = L63TrainingData(
                crop_T=10_000,
                data_folder=getattr(args, 'l63_data_val', None),
                train_size=eval_size,
                train_operator=True,
                validation=True,
                noisy_scale=noisy_scale,
            )
            eval_name = f'validation_on_noise_data_{_noise_tag(noisy_scale)}'
            result_filename = 'Results_validation_on_noise_data.txt'
        else:
            eval_size = int(getattr(args, 'l63_eval_size_test', 20))
            initial_dataset = L63TestingData(
                data_folder=getattr(args, 'l63_data_test', None),
                noisy_scale=noisy_scale,
            )
            eval_dataset = L63TestingData(
                data_folder=getattr(args, 'l63_data_test', None),
                noisy_scale=noisy_scale,
            )
            eval_name = f'test_on_noise_data_{_noise_tag(noisy_scale)}'
            result_filename = 'Results_test_on_noise_data.txt'

    all_data_initial = _get_all_data(initial_dataset)
    all_data_eval = _get_all_data(eval_dataset)

    params_initial, trajs_initial = all_data_initial[0], all_data_initial[1]
    params_eval, trajs_eval = all_data_eval[0], all_data_eval[1]

    eval_size = min(eval_size, len(trajs_eval))
    params_initial = params_initial[:eval_size]
    trajs_initial = trajs_initial[:eval_size]
    params_eval = params_eval[:eval_size]
    trajs_eval = trajs_eval[:eval_size]

    state_dim = trajs_eval.shape[-1]
    total_t = trajs_eval.shape[1]

    eval_root = output_path if output_path else '.'
    rollout_dir = f'rollout_{x_len}'
    eval_dir = os.path.join(eval_root, 'eval_noisy_trainval_clean_test', rollout_dir, eval_name)
    os.makedirs(eval_dir, exist_ok=True)
    approach_label = _resolve_l63_approach_label(args)

    # -------------------------------------------------------------------------
    # Multi-step rollout and attractor diagnostics
    # -------------------------------------------------------------------------
    x_0_start = 50
    x_end = x_0_start + x_len

    geometry_l1_scores = []
    rel_l2_rollout_scores = []
    attractor_metric_values = {
        'psd_log_l1': [],
        'psd_jsd': [],
        'autocorr_l1': [],
        'regime_occupancy_abs': [],
        'switch_rate_abs': [],
    }

    with torch.no_grad():
        operator.eval()

        x0 = trajs_initial[:, x_0_start, :][:, None, :].to(args.gpu)  # (B, 1, 3)
        rollout = [x0]

        # Keep the same anchored-vs-free-running convention as KS.
        if x_len == int(args.x_len):
            for t in range(1, x_len):
                if t % 2 == 0:
                    x_next = trajs_initial[:, x_0_start + t, :][:, None, :].to(args.gpu)
                else:
                    x_next = operator(x0)
                rollout.append(x_next)
                x0 = x_next
        else:
            for _ in range(1, x_len):
                x_next = operator(x0)
                rollout.append(x_next)
                x0 = x_next

        out_traj = torch.cat(rollout, dim=1)  # (B, x_len, 3)
        true_traj = trajs_eval[:, x_0_start:x_end, :].to(args.gpu)

        compare_len = min(out_traj.shape[1], true_traj.shape[1])
        if compare_len < x_len:
            print(
                f'[eval_l63] WARNING: true trajectory shorter than requested rollout '
                f'(T={total_t}, x_0_start={x_0_start}, x_len={x_len}). '
                f'Using compare_len={compare_len}.'
            )

        out_traj = out_traj[:, :compare_len, :]
        true_traj = true_traj[:, :compare_len, :]

        n_plot = int(getattr(args, 'l63_plot_samples', 3))
        plot_t = compare_len
        _save_l63_rollout_plots(
            true_traj[:, :plot_t],
            out_traj[:, :plot_t],
            output_path=eval_dir,
            max_samples=n_plot,
            approach_label=approach_label,
        )

        for i in range(eval_size):
            u_true = true_traj[i]
            u_pred = out_traj[i]

            rel_l2_rollout_scores.append(LpLoss_(2).rel(u_pred[None], u_true[None]).cpu().item())
            geometry_l1_scores.append(
                cal_stats_l1_score_l63(u_true.cpu().numpy(), u_pred.cpu().numpy())
            )
            extra_metrics = cal_l63_attractor_metrics(
                u_true.cpu().numpy(),
                u_pred.cpu().numpy(),
                max_lag=int(getattr(args, 'l63_attractor_max_lag', 64)),
            )
            for key in attractor_metric_values:
                attractor_metric_values[key].append(float(extra_metrics[key]))

    rollout_l2_stats = {
        'q25': float(np.quantile(rel_l2_rollout_scores, 0.25)),
        'q50': float(np.quantile(rel_l2_rollout_scores, 0.50)),
        'q75': float(np.quantile(rel_l2_rollout_scores, 0.75)),
    }
    geometry_stats = {
        'q25': float(np.quantile(geometry_l1_scores, 0.25)),
        'q50': float(np.quantile(geometry_l1_scores, 0.50)),
        'q75': float(np.quantile(geometry_l1_scores, 0.75)),
    }
    attractor_stats = {
        key: {
            'q25': float(np.quantile(values, 0.25)),
            'q50': float(np.quantile(values, 0.50)),
            'q75': float(np.quantile(values, 0.75)),
            'mean': float(np.mean(values)),
        }
        for key, values in attractor_metric_values.items()
    }

    print(
        f'Rollout relative L2 (q25/q50/q75): '
        f'{rollout_l2_stats["q25"]:.6f} / {rollout_l2_stats["q50"]:.6f} / {rollout_l2_stats["q75"]:.6f}'
    )
    print(
        f'Attractor geometry L1 (q25/q50/q75): '
        f'{geometry_stats["q25"]:.6f} / {geometry_stats["q50"]:.6f} / {geometry_stats["q75"]:.6f}'
    )
    print(
        'Attractor diagnostics (mean): '
        f'PSD={attractor_stats["psd_log_l1"]["mean"]:.6f} | '
        f'PSD-JSD={attractor_stats["psd_jsd"]["mean"]:.6f} | '
        f'ACF={attractor_stats["autocorr_l1"]["mean"]:.6f} | '
        f'Occ={attractor_stats["regime_occupancy_abs"]["mean"]:.6f} | '
        f'Switch={attractor_stats["switch_rate_abs"]["mean"]:.6f}'
    )

    # -------------------------------------------------------------------------
    # One-step RMSE
    # -------------------------------------------------------------------------
    eval_bsize = 5
    l2_one_step = []

    with torch.no_grad():
        operator.eval()
        for s in range(0, eval_size, eval_bsize):
            e = min(s + eval_bsize, eval_size)
            traj_initial_b = trajs_initial[s:e]
            traj_eval_b = trajs_eval[s:e]

            # Evaluate one-step prediction over full available stream.
            x_now = traj_initial_b[:, :-1].reshape(-1, 1, state_dim).to(args.gpu)
            x_next_true = traj_eval_b[:, 1:].reshape(-1, state_dim).to(args.gpu)
            x_next_pred = operator(x_now).squeeze(1)

            l2_one_step.append(LpLoss_(2).rel(x_next_pred, x_next_true).cpu().item())

    if not l2_one_step:
        raise ValueError('No one-step L63 evaluation samples were processed.')

    rmse = float(np.mean(l2_one_step))
    rmse_array = np.asarray(l2_one_step, dtype=float)

    print(f'One-step relative L2 (RMSE): {rmse:.6f}')

    # -------------------------------------------------------------------------
    # Persist results
    # -------------------------------------------------------------------------
    result_path = os.path.join(eval_dir, result_filename)
    with open(result_path, 'w') as f:
        f.write(f'noise {noisy_scale} with eval length {x_len} training length {args.x_len} \n')
        f.write(
            f"mse_['rMSE', {np.quantile(rmse_array, 0.25)}, '50 percentile', "
            f"{np.quantile(rmse_array, 0.50)}, {np.quantile(rmse_array, 0.75)}] \n"
        )
        f.write(
            'rollout_rel_l2: '
            f"{[np.mean(rel_l2_rollout_scores), np.quantile(rel_l2_rollout_scores, 0.25), '50 percentile', np.quantile(rel_l2_rollout_scores, 0.50), np.quantile(rel_l2_rollout_scores, 0.75)]} \n"
        )
        f.write(
            'attractor_geometry_l1: '
            f"{[np.mean(geometry_l1_scores), np.quantile(geometry_l1_scores, 0.25), '50 percentile', np.quantile(geometry_l1_scores, 0.50), np.quantile(geometry_l1_scores, 0.75)]} \n"
        )
        f.write(
            'attractor_psd_log_l1: '
            f"{[attractor_stats['psd_log_l1']['mean'], attractor_stats['psd_log_l1']['q25'], '50 percentile', attractor_stats['psd_log_l1']['q50'], attractor_stats['psd_log_l1']['q75']]} \n"
        )
        f.write(
            'attractor_psd_jsd: '
            f"{[attractor_stats['psd_jsd']['mean'], attractor_stats['psd_jsd']['q25'], '50 percentile', attractor_stats['psd_jsd']['q50'], attractor_stats['psd_jsd']['q75']]} \n"
        )
        f.write(
            'attractor_autocorr_l1: '
            f"{[attractor_stats['autocorr_l1']['mean'], attractor_stats['autocorr_l1']['q25'], '50 percentile', attractor_stats['autocorr_l1']['q50'], attractor_stats['autocorr_l1']['q75']]} \n"
        )
        f.write(
            'attractor_regime_occupancy_abs: '
            f"{[attractor_stats['regime_occupancy_abs']['mean'], attractor_stats['regime_occupancy_abs']['q25'], '50 percentile', attractor_stats['regime_occupancy_abs']['q50'], attractor_stats['regime_occupancy_abs']['q75']]} \n"
        )
        f.write(
            'attractor_switch_rate_abs: '
            f"{[attractor_stats['switch_rate_abs']['mean'], attractor_stats['switch_rate_abs']['q25'], '50 percentile', attractor_stats['switch_rate_abs']['q50'], attractor_stats['switch_rate_abs']['q75']]} \n"
        )

    if noisy_scale == 0:
        rollout_result_path = os.path.join(eval_root, f'Results_l63_test_rollout_{x_len}.txt')
        with open(rollout_result_path, 'w') as f:
            f.write(f'L63 evaluation | rollout_len={x_len} | state_dim={state_dim}\n\n')
            f.write(f'One-step relative L2 (RMSE): {rmse:.6f}\n')
            f.write(
                'Rollout relative L2 q25/q50/q75: '
                f'{rollout_l2_stats["q25"]:.6f} / {rollout_l2_stats["q50"]:.6f} / {rollout_l2_stats["q75"]:.6f}\n'
            )
            f.write(
                'Attractor geometry L1 q25/q50/q75: '
                f'{geometry_stats["q25"]:.6f} / {geometry_stats["q50"]:.6f} / {geometry_stats["q75"]:.6f}\n'
            )
            f.write(
                'Attractor PSD-log-L1 q25/q50/q75: '
                f'{attractor_stats["psd_log_l1"]["q25"]:.6f} / '
                f'{attractor_stats["psd_log_l1"]["q50"]:.6f} / '
                f'{attractor_stats["psd_log_l1"]["q75"]:.6f}\n'
            )
            f.write(
                'Attractor PSD-JSD q25/q50/q75: '
                f'{attractor_stats["psd_jsd"]["q25"]:.6f} / '
                f'{attractor_stats["psd_jsd"]["q50"]:.6f} / '
                f'{attractor_stats["psd_jsd"]["q75"]:.6f}\n'
            )
            f.write(
                'Attractor autocorr-L1 q25/q50/q75: '
                f'{attractor_stats["autocorr_l1"]["q25"]:.6f} / '
                f'{attractor_stats["autocorr_l1"]["q50"]:.6f} / '
                f'{attractor_stats["autocorr_l1"]["q75"]:.6f}\n'
            )
            f.write(
                'Regime occupancy abs q25/q50/q75: '
                f'{attractor_stats["regime_occupancy_abs"]["q25"]:.6f} / '
                f'{attractor_stats["regime_occupancy_abs"]["q50"]:.6f} / '
                f'{attractor_stats["regime_occupancy_abs"]["q75"]:.6f}\n'
            )
            f.write(
                'Switch-rate abs q25/q50/q75: '
                f'{attractor_stats["switch_rate_abs"]["q25"]:.6f} / '
                f'{attractor_stats["switch_rate_abs"]["q50"]:.6f} / '
                f'{attractor_stats["switch_rate_abs"]["q75"]:.6f}\n'
            )

        eval_lengths = [int(v) for v in getattr(args, 'l63_eval_lengths', [x_len])]
        if x_len == max(eval_lengths):
            legacy_result_path = os.path.join(eval_root, 'Results_l63_test.txt')
            with open(legacy_result_path, 'w') as f:
                f.write(f'L63 evaluation | rollout_len={x_len} | state_dim={state_dim}\n\n')
                f.write(f'One-step relative L2 (RMSE): {rmse:.6f}\n')
                f.write(
                    'Rollout relative L2 q25/q50/q75: '
                    f'{rollout_l2_stats["q25"]:.6f} / {rollout_l2_stats["q50"]:.6f} / {rollout_l2_stats["q75"]:.6f}\n'
                )
                f.write(
                    'Attractor geometry L1 q25/q50/q75: '
                    f'{geometry_stats["q25"]:.6f} / {geometry_stats["q50"]:.6f} / {geometry_stats["q75"]:.6f}\n'
                )
                f.write(
                    'Attractor PSD-log-L1 q25/q50/q75: '
                    f'{attractor_stats["psd_log_l1"]["q25"]:.6f} / '
                    f'{attractor_stats["psd_log_l1"]["q50"]:.6f} / '
                    f'{attractor_stats["psd_log_l1"]["q75"]:.6f}\n'
                )
                f.write(
                    'Attractor PSD-JSD q25/q50/q75: '
                    f'{attractor_stats["psd_jsd"]["q25"]:.6f} / '
                    f'{attractor_stats["psd_jsd"]["q50"]:.6f} / '
                    f'{attractor_stats["psd_jsd"]["q75"]:.6f}\n'
                )
                f.write(
                    'Attractor autocorr-L1 q25/q50/q75: '
                    f'{attractor_stats["autocorr_l1"]["q25"]:.6f} / '
                    f'{attractor_stats["autocorr_l1"]["q50"]:.6f} / '
                    f'{attractor_stats["autocorr_l1"]["q75"]:.6f}\n'
                )
                f.write(
                    'Regime occupancy abs q25/q50/q75: '
                    f'{attractor_stats["regime_occupancy_abs"]["q25"]:.6f} / '
                    f'{attractor_stats["regime_occupancy_abs"]["q50"]:.6f} / '
                    f'{attractor_stats["regime_occupancy_abs"]["q75"]:.6f}\n'
                )
                f.write(
                    'Switch-rate abs q25/q50/q75: '
                    f'{attractor_stats["switch_rate_abs"]["q25"]:.6f} / '
                    f'{attractor_stats["switch_rate_abs"]["q50"]:.6f} / '
                    f'{attractor_stats["switch_rate_abs"]["q75"]:.6f}\n'
                )

    return rmse, rollout_l2_stats, geometry_stats
