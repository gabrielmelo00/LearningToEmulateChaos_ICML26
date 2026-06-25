"""
Lorenz-63 visualization helpers for rollout and attractor geometry comparison.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def _as_l63_traj(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f'{name} must have shape (T,3), got {arr.shape}')
    return arr


def _downsample_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, max_points).astype(int)


def plot_l63_rollout_comparison(
    traj_true: np.ndarray,
    traj_pred: np.ndarray,
    title: str = 'Lorenz-63 Rollout Comparison',
    max_points_3d: int = 20_000,
    approach_label: str | None = None,
):
    """
    3D attractor overlay + component-wise time-series comparison.
    """
    true = _as_l63_traj(traj_true, 'traj_true')
    pred = _as_l63_traj(traj_pred, 'traj_pred')

    t = np.arange(min(len(true), len(pred)))
    true = true[: len(t)]
    pred = pred[: len(t)]

    idx = _downsample_indices(len(t), max_points=max_points_3d)

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(3, 2, hspace=0.30)

    ax3d = fig.add_subplot(gs[:, 0], projection='3d')
    ax3d.plot(true[idx, 0], true[idx, 1], true[idx, 2], color='steelblue', lw=0.8, label='Ground truth')
    ax3d.plot(pred[idx, 0], pred[idx, 1], pred[idx, 2], color='darkorange', lw=0.8, label='Emulator')
    ax3d.set_xlabel('x')
    ax3d.set_ylabel('y')
    ax3d.set_zlabel('z')
    ax3d.legend(frameon=False, fontsize=9)

    labels = ['x', 'y', 'z']
    colors = ['tab:blue', 'tab:green', 'tab:red']
    for i in range(3):
        ax = fig.add_subplot(gs[i, 1])
        ax.plot(t, true[:, i], color=colors[i], lw=1.0, label=f'{labels[i]} true')
        ax.plot(t, pred[:, i], color=colors[i], lw=1.0, ls='--', alpha=0.9, label=f'{labels[i]} pred')
        ax.set_ylabel(labels[i])
        ax.grid(True, ls='--', alpha=0.3)
        if i < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('time index')
        ax.legend(loc='upper right', fontsize=8, frameon=False)

    if approach_label:
        fig.suptitle(f'{title} | {approach_label}', y=0.98)
    else:
        fig.suptitle(title, y=0.98)
    return fig


def plot_l63_attractor_geometry_comparison(
    traj_true: np.ndarray,
    traj_pred: np.ndarray,
    max_points_3d: int = 30_000,
    approach_label: str | None = None,
    show_titles: bool = True,
):
    """
    3D attractor overlay (ground truth vs emulator).
    """
    true = _as_l63_traj(traj_true, 'traj_true')
    pred = _as_l63_traj(traj_pred, 'traj_pred')

    n = min(len(true), len(pred))
    true = true[:n]
    pred = pred[:n]

    idx = _downsample_indices(n, max_points=max_points_3d)

    fig = plt.figure(figsize=(8, 6))
    ax3d = fig.add_subplot(1, 1, 1, projection='3d')
    ax3d.plot(true[idx, 0], true[idx, 1], true[idx, 2], color='steelblue', lw=0.7, label='Ground truth')
    ax3d.plot(pred[idx, 0], pred[idx, 1], pred[idx, 2], color='darkorange', lw=0.7, label='Emulator')
    ax3d.set_xlabel('x')
    ax3d.set_ylabel('y')
    ax3d.set_zlabel('z')
    if show_titles:
        ax3d.set_title('3D Attractor Overlay')
    ax3d.legend(frameon=False, fontsize=8)

    if show_titles:
        if approach_label:
            fig.suptitle(f'Lorenz-63 Attractor Geometry Comparison | {approach_label}', y=0.98)
        else:
            fig.suptitle('Lorenz-63 Attractor Geometry Comparison', y=0.98)
    return fig


def plot_l63_projections_colored_by_summary(
    u_true: np.ndarray,
    s_true: np.ndarray,
    max_points: int | None = 10_000,
    cmap: str = 'viridis',
    cbar_label: str = 'summary',
    vmin: float | None = None,
    vmax: float | None = None,
):
    """
    Three 2D projections (x,y), (x,z), (y,z) colored by a scalar summary.
    """
    u_true = _as_l63_traj(u_true, 'u_true')
    s = np.asarray(s_true).reshape(-1)
    if s.shape[0] != u_true.shape[0]:
        raise ValueError(f's_true length mismatch: {s.shape[0]} vs {u_true.shape[0]}')

    if max_points is not None and len(s) > max_points:
        idx = _downsample_indices(len(s), max_points)
        u_true = u_true[idx]
        s = s[idx]

    x, y, z = u_true[:, 0], u_true[:, 1], u_true[:, 2]

    if vmin is None:
        vmin = float(np.min(s))
    if vmax is None:
        vmax = float(np.max(s))

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    sc0 = axs[0].scatter(x, y, c=s, s=6, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('(x, y)')

    axs[1].scatter(x, z, c=s, s=6, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('z')
    axs[1].set_title('(x, z)')

    axs[2].scatter(y, z, c=s, s=6, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[2].set_xlabel('y')
    axs[2].set_ylabel('z')
    axs[2].set_title('(y, z)')

    fig.colorbar(sc0, ax=axs, fraction=0.03, pad=0.02, label=cbar_label)
    return fig
