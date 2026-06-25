import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_batch_gradient(input, wrt='T', order=1):
    input = input.clone()
    assert len(input.shape) == 3  # B x T x d
    bsz, t_len, dim = input.shape
    if wrt == 'T':
        ans = input.permute(0, 2, 1).reshape(bsz * dim, t_len)
        grad = torch.gradient(ans, dim=1)[0]
        if order > 1:
            grad = torch.gradient(grad, dim=1)[0]
        grad = grad.reshape(bsz, dim, t_len).permute(0, 2, 1)
    elif wrt == 'd':
        ans = input.reshape(bsz * t_len, dim)
        grad = torch.gradient(ans, dim=1)[0]
        if order > 1:
            grad = torch.gradient(grad, dim=1)[0]
        grad = grad.reshape(bsz, t_len, dim)
    return grad


class SummaryNet(nn.Module):
    def __init__(self, summary_dim=3, hidden_dim=128, mode='physics', state_dim=60,
                 window_size=2):
        """
        Args:
            summary_dim:  dimension of output summary statistics
            hidden_dim:   hidden dimension for MLP layers
            mode:         one of 'physics', 'ks_physics', 'pointwise',
                          'local', 'statewise', 'linear'
            state_dim:    spatial dimension of state (used only by 'statewise' mode)
            window_size:  half-window for 'local' mode; input width = 2*window_size+1
        """
        super().__init__()
        self.summary_dim = summary_dim
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.state_dim = state_dim
        self.window_size = window_size

        if mode == 'physics':
            # L96-style physics: project 3 handcrafted stats with a linear map
            self.proj = nn.Linear(3, summary_dim)

        elif mode == 'ks_physics':
            # KS-physics: nonlinear projection of (du/dt, du/dx, d²u/dx²).
            # Strictly more expressive than fixed OT (which is the identity in R³).
            # With summary_dim > 3 it lifts the physics features into a richer space.
            self.proj = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, summary_dim),
            )

        elif mode == 'pointwise':
            # Data-driven MLP: maps each scalar u(x,t) independently → R^d.
            # Can only match marginal value histograms; no spatial structure.
            self.mlp = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, summary_dim),
            )

        elif mode == 'local':
            # Spatial-neighborhood MLP: maps a window of (2*window_size+1) spatially
            # adjacent values → R^d.  Uses periodic (circular) padding matching the
            # KS periodic boundary conditions.  The MLP implicitly learns finite-
            # difference approximations to spatial derivatives, without needing to
            # specify which order.
            in_dim = 2 * window_size + 1
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, summary_dim),
            )

        elif mode == 'statewise':
            # Full-state MLP: maps entire spatial snapshot u(·,t) → per-point R^d.
            # Most expressive but ~130K parameters trained with B=5 — prone to
            # overfitting / collapse.
            self.mlp = nn.Sequential(
                nn.Linear(state_dim - 4, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, (state_dim - 4) * summary_dim),
            )

        elif mode == 'linear':
            # Learnable linear map from full state snapshot to summary.
            # For L63 this is exactly a linear projection from R^3 to R^summary_dim.
            self.proj = nn.Linear(state_dim, summary_dim)

        else:
            raise ValueError(
                f"Invalid mode: {mode}. "
                "Choose 'ks_physics', 'local', 'pointwise', 'physics', 'statewise', or 'linear'"
            )

    def forward(self, traj):
        """
        Parameters
        ----------
        traj : torch.Tensor, shape (B, T, N)

        Returns
        -------
        torch.Tensor, shape (B, n_points, summary_dim)
            OT point cloud — one point per (t, x) position after boundary trimming.
        """
        if self.mode == 'physics':
            # L96 advection-based statistics: u_{k-1}*(u_{k-2}-u_{k+1}), du/dt, u.
            # Designed for L96; the neighbor-product term has no direct KS meaning,
            # but it still encodes local spatial structure and can work incidentally.
            var = traj
            var_k_1   = torch.roll(var, 1,  dims=2)
            var_k_2   = torch.roll(var, 2,  dims=2)
            var_k_p_1 = torch.roll(var, -1, dims=2)
            grad_t = _compute_batch_gradient(var, wrt='T', order=1)
            advection_stats = var_k_1 * (var_k_2 - var_k_p_1)
            advection_stats = advection_stats[:, 2:-2, 2:-2]
            grad_t = grad_t[:, 2:-2, 2:-2]
            var    = var[:, 2:-2, 2:-2]
            stats  = torch.stack([advection_stats, grad_t, var], dim=-1)
            bsz, t_len, dim, _ = stats.shape
            stats = self.proj(stats.reshape(-1, 3)).reshape(bsz, t_len, dim, self.summary_dim)
            return stats.reshape(bsz, -1, self.summary_dim)

        elif self.mode == 'ks_physics':
            # Compute (u, u*du/dx, du/dt) — mirrors cal_stats_ks exactly.
            # Spatial derivative uses FFT (exact for KS periodic BCs).
            # Only time boundaries are trimmed; spatial dim is fully periodic.
            from scripts.cal_stats_ks import _spectral_dx_batch, _batch_time_derivative
            B, T, N = traj.shape

            du_dx = _spectral_dx_batch(traj)            # (B, T, N) — exact, periodic
            advec = traj * du_dx                        # (B, T, N) — u * du/dx
            du_dt = _batch_time_derivative(traj)        # (B, T, N)

            trim = 2
            u_tr     = traj [:, trim:-trim, :]          # (B, T-4, N)
            advec_tr = advec[:, trim:-trim, :]
            du_dt_tr = du_dt[:, trim:-trim, :]

            # stack → (B, (T-4)*N, 3) then project
            stats = torch.stack(
                [u_tr.reshape(B, -1),
                 advec_tr.reshape(B, -1),
                 du_dt_tr.reshape(B, -1)],
                dim=-1
            )
            return self.proj(stats.reshape(-1, 3)).reshape(B, -1, self.summary_dim)

        elif self.mode == 'pointwise':
            bsz, t_len, dim = traj.shape
            traj_trimmed = traj[:, 2:-2, 2:-2]
            bsz, t_trim, d_trim = traj_trimmed.shape
            points  = traj_trimmed.reshape(-1, 1)
            summary = self.mlp(points)
            return summary.reshape(bsz, -1, self.summary_dim)

        elif self.mode == 'local':
            # Trim time boundary, keep full spatial dimension (periodic padding handles edges)
            bsz, t_len, N = traj.shape
            t_trim_traj = traj[:, 2:-2, :]          # (B, T-4, N)
            bsz, T_t, N = t_trim_traj.shape
            w = self.window_size
            # circular pad to respect KS periodic BCs
            padded  = F.pad(t_trim_traj, (w, w), mode='circular')  # (B, T-4, N+2w)
            patches = padded.unfold(2, 2 * w + 1, 1)               # (B, T-4, N, 2w+1)
            flat    = patches.reshape(-1, 2 * w + 1)
            return self.mlp(flat).reshape(bsz, -1, self.summary_dim)

        elif self.mode == 'statewise':
            bsz, t_len, dim = traj.shape
            traj_trimmed = traj[:, 2:-2, 2:-2]
            bsz, t_trim, d_trim = traj_trimmed.shape
            states  = traj_trimmed.reshape(bsz * t_trim, d_trim)
            summary = self.mlp(states)
            summary = summary.reshape(bsz, t_trim, d_trim, self.summary_dim)
            return summary.reshape(bsz, -1, self.summary_dim)

        elif self.mode == 'linear':
            bsz, t_len, d = traj.shape
            traj_trimmed = traj[:, 2:-2, :] if t_len > 4 else traj
            bsz, t_trim, d = traj_trimmed.shape
            states = traj_trimmed.reshape(bsz * t_trim, d)
            summary = self.proj(states)
            return summary.reshape(bsz, t_trim, self.summary_dim)


class Critic(nn.Module):
    def __init__(self, summary_dim=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, stats):
        # stats: (B, N, D)
        out = self.net(stats)
        return out.mean(dim=1).squeeze(-1)
