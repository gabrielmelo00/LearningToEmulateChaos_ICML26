"""
scripts/cal_stats_ks.py
-----------------------
Statistics and distributional metrics for the KS experiment.
Equivalent to scripts/cal_stats_l96.py in the L96 pipeline.

Statistics computed per trajectory:
    - du/dt   : temporal derivative  (finite difference along time axis)
    - du/dx   : first spatial derivative  (spectral, via torch.gradient)
    - d²u/dx² : second spatial derivative (spectral, via torch.gradient)

These three quantities capture the dominant dynamical balance in KS:
    u_t = -u*u_x - u_xx - u_xxxx
and together form a physically meaningful 3D joint distribution for
comparing truth vs. emulator output.

The L1 score between the joint 3D histograms (chi-score) mirrors the
metric used for L96 in cal_stats_l96.py.
"""

import numpy as np
import torch


# ─────────────────────────────────────────────
# Batch derivative utilities
# ─────────────────────────────────────────────

def _batch_time_derivative(traj, order=1):
    """
    Compute temporal derivative along dim=1 (time axis) for a batch.

    Parameters
    ----------
    traj : torch.Tensor, shape (B, T, N)
    order : int, 1 or 2

    Returns
    -------
    grad : torch.Tensor, shape (B, T, N)
    mask : torch.Tensor, shape (B, T, N)  — zeros at boundary time steps
    """
    B, T, N = traj.shape
    flat = traj.permute(0, 2, 1).reshape(B * N, T)   # (B*N, T)
    grad = torch.gradient(flat, dim=1)[0]
    if order > 1:
        grad = torch.gradient(grad, dim=1)[0]
    grad = grad.reshape(B, N, T).permute(0, 2, 1)    # (B, T, N)

    mask = torch.ones(B, T, N, dtype=traj.dtype)
    mask[:, :order,  :] = 0
    mask[:, -order:, :] = 0
    return grad, mask


def _batch_space_derivative(traj, order=1):
    """
    Compute spatial derivative along dim=2 (space axis) for a batch.

    Parameters
    ----------
    traj : torch.Tensor, shape (B, T, N)
    order : int, 1 or 2

    Returns
    -------
    grad : torch.Tensor, shape (B, T, N)
    mask : torch.Tensor, shape (B, T, N)  — zeros at boundary spatial points
    """
    B, T, N = traj.shape
    flat = traj.reshape(B * T, N)                     # (B*T, N)
    grad = torch.gradient(flat, dim=1)[0]
    if order > 1:
        grad = torch.gradient(grad, dim=1)[0]
    grad = grad.reshape(B, T, N)

    mask = torch.ones(B, T, N, dtype=traj.dtype)
    mask[:, :, :order]  = 0
    mask[:, :, -order:] = 0
    return grad, mask


# ─────────────────────────────────────────────
# Batch stats assembler  (used during training)
# ─────────────────────────────────────────────

def cal_stats_ks(anchor_t, out_t):
    """
    Compute (du/dt, du/dx, d²u/dx²) for truth and prediction tensors.
    Equivalent to cal_stats_l96 in the L96 pipeline.

    Parameters
    ----------
    anchor_t : torch.Tensor, shape (B, T, N)   — ground truth trajectories
    out_t    : torch.Tensor, shape (B, T, N)   — emulator output trajectories

    Returns
    -------
    anchor_stats : torch.Tensor, shape (B, T*N, 3)
    out_stats    : torch.Tensor, shape (B, T*N, 3)
    """
    du_dt,   _    = _batch_time_derivative(anchor_t,  order=1)
    du_dx,   _    = _batch_space_derivative(anchor_t, order=1)
    d2u_dx2, mask = _batch_space_derivative(anchor_t, order=2)

    trim = 2
    du_dt   = du_dt  [:, trim:-trim, trim:-trim]
    du_dx   = du_dx  [:, trim:-trim, trim:-trim]
    d2u_dx2 = d2u_dx2[:, trim:-trim, trim:-trim]
    B       = anchor_t.shape[0]

    anchor_stats = torch.stack(
        [du_dt.reshape(B, -1),
         du_dx.reshape(B, -1),
         d2u_dx2.reshape(B, -1)], dim=-1
    )                                               # (B, (T-4)*(N-4), 3)

    du_dt_o,   _    = _batch_time_derivative(out_t,  order=1)
    du_dx_o,   _    = _batch_space_derivative(out_t, order=1)
    d2u_dx2_o, _    = _batch_space_derivative(out_t, order=2)

    du_dt_o   = du_dt_o  [:, trim:-trim, trim:-trim]
    du_dx_o   = du_dx_o  [:, trim:-trim, trim:-trim]
    d2u_dx2_o = d2u_dx2_o[:, trim:-trim, trim:-trim]

    out_stats = torch.stack(
        [du_dt_o.reshape(B, -1),
         du_dx_o.reshape(B, -1),
         d2u_dx2_o.reshape(B, -1)], dim=-1
    )                                               # (B, (T-4)*(N-4), 3)

    return anchor_stats, out_stats


# ─────────────────────────────────────────────
# L1 score on joint 3D histogram  (used during evaluation)
# ─────────────────────────────────────────────

def cal_stats_l1_score_ks(anchor_t, out_t):
    """
    Compute the L1 (chi) score between the joint 3D histograms of
    (du/dt, du/dx, d²u/dx²) for truth vs. emulator output.

    Equivalent to cal_stats_l1_score in cal_stats_l96.py.

    Parameters
    ----------
    anchor_t : np.ndarray, shape (T, N)   — ground truth trajectory
    out_t    : np.ndarray, shape (T, N)   — emulator output trajectory

    Returns
    -------
    l1_score_3d : float
        L1 distance between normalised 3D histograms. Lower is better.
    """

    def _numpy_stats(u):
        """Return (du/dt, du/dx, d²u/dx²) as flat numpy arrays."""
        u_t = torch.from_numpy(u)                    # (T, N)

        flat_t = u_t.T                               # (N, T)
        dt_grad = torch.gradient(flat_t, dim=1)[0].T  # (T, N)

        flat_x  = u_t                                # (T, N)
        dx_grad  = torch.gradient(flat_x, dim=1)[0]  # (T, N)
        d2x_grad = torch.gradient(dx_grad, dim=1)[0] # (T, N)

        trim = 2
        dt_grad  = dt_grad[trim:-trim,  trim:-trim]
        dx_grad  = dx_grad[trim:-trim,  trim:-trim]
        d2x_grad = d2x_grad[trim:-trim, trim:-trim]

        return (dt_grad.numpy().reshape(-1),
                dx_grad.numpy().reshape(-1),
                d2x_grad.numpy().reshape(-1))

    def _l1(hist_truth, hist_pred):
        h_truth = hist_truth[0].reshape(-1)
        h_pred  = hist_pred[0].reshape(-1)
        norm    = h_truth.sum()
        return float(np.abs(h_truth / norm - h_pred / norm).sum())

    dt_a,  dx_a,  d2x_a  = _numpy_stats(anchor_t)
    dt_o,  dx_o,  d2x_o  = _numpy_stats(out_t)

    anchor_stats = np.stack([dt_a,  dx_a,  d2x_a],  axis=1)
    out_stats    = np.stack([dt_o,  dx_o,  d2x_o],  axis=1)

    n_pts      = anchor_stats.shape[0]
    bins_per_dim = max(5, int(np.floor(n_pts**(1/3))))

    hist_truth = np.histogramdd(anchor_stats, bins=bins_per_dim)
    hist_pred  = np.histogramdd(out_stats,    bins=hist_truth[1])  # same edges

    return _l1(hist_truth, hist_pred)