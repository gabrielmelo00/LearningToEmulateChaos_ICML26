"""
Statistics for Lorenz-63 OT losses and evaluation.

Design choice for fixed OT:
- Use configurable state coordinates for fixed OT summary statistics.

For rollout geometry diagnostics:
- Compare the full 3D attractor occupancy using histogram L1 distance.
"""

from __future__ import annotations

import numpy as np
import torch


def _validate_l63_batch(anchor_t: torch.Tensor, out_t: torch.Tensor) -> None:
    if anchor_t.ndim != 3 or anchor_t.shape[-1] != 3:
        raise ValueError(f'anchor_t must have shape (B,T,3), got {tuple(anchor_t.shape)}')
    if out_t.ndim != 3 or out_t.shape[-1] != 3:
        raise ValueError(f'out_t must have shape (B,T,3), got {tuple(out_t.shape)}')


def _resolve_dims(dims: tuple[int, ...] | list[int] | None) -> tuple[int, ...]:
    if dims is None:
        return (0, 1, 2)
    dims_tuple = tuple(int(d) for d in dims)
    if len(dims_tuple) == 0:
        raise ValueError('dims cannot be empty')
    invalid = [d for d in dims_tuple if d not in (0, 1, 2)]
    if invalid:
        raise ValueError(f'Invalid L63 dims {invalid}; allowed dims are 0, 1, 2.')
    return dims_tuple


def _resolve_lags(lags: tuple[int, ...] | list[int] | None) -> tuple[int, ...]:
    if lags is None:
        return (0,)
    lags_tuple = tuple(int(l) for l in lags)
    if len(lags_tuple) == 0:
        raise ValueError('lags cannot be empty')
    invalid = [l for l in lags_tuple if l < 0]
    if invalid:
        raise ValueError(f'Invalid negative lags: {invalid}')
    return lags_tuple


def cal_stats_l63(
    anchor_t: torch.Tensor,
    out_t: torch.Tensor,
    dims: tuple[int, ...] | list[int] | None = None,
    feature_mode: str = 'state',
    lags: tuple[int, ...] | list[int] | None = None,
    trim: int = 2,
):
    """
    Fixed OT summary for Lorenz-63: use configurable coordinates.

    Parameters
    ----------
    anchor_t : torch.Tensor, shape (B, T, 3)
    out_t    : torch.Tensor, shape (B, T, 3)

    Returns
    -------
    anchor_stats : torch.Tensor, shape (B, T_trim, D)
    out_stats    : torch.Tensor, shape (B, T_trim, D)
    """
    _validate_l63_batch(anchor_t, out_t)
    dims_tuple = _resolve_dims(dims)
    lags_tuple = _resolve_lags(lags)

    trim = int(trim)
    trim = trim if (trim > 0 and anchor_t.shape[1] > (2 * trim)) else 0
    if trim > 0:
        anchor_base = anchor_t[:, trim:-trim, :][:, :, dims_tuple]
        out_base = out_t[:, trim:-trim, :][:, :, dims_tuple]
    else:
        anchor_base = anchor_t[:, :, :][:, :, dims_tuple]
        out_base = out_t[:, :, :][:, :, dims_tuple]

    if feature_mode == 'state':
        anchor_stats = anchor_base
        out_stats = out_base
    elif feature_mode == 'lagged_state':
        max_lag = max(lags_tuple)
        if anchor_base.shape[1] <= max_lag:
            raise ValueError(
                f'Not enough timesteps for max lag {max_lag}: got T={anchor_base.shape[1]}'
            )
        t_eff = anchor_base.shape[1] - max_lag
        anchor_feats = []
        out_feats = []
        for lag in lags_tuple:
            start = max_lag - lag
            end = start + t_eff
            anchor_feats.append(anchor_base[:, start:end, :])
            out_feats.append(out_base[:, start:end, :])
        anchor_stats = torch.cat(anchor_feats, dim=-1)
        out_stats = torch.cat(out_feats, dim=-1)
    else:
        raise ValueError(
            f'Unknown feature_mode={feature_mode!r}. '
            "Expected one of ['state', 'lagged_state']."
        )

    return anchor_stats.contiguous(), out_stats.contiguous()


def cal_stats_l1_score_l63(
    anchor_t: np.ndarray,
    out_t: np.ndarray,
    bins: int = 24,
) -> float:
    """
    Attractor geometry distance using 3D histogram L1.

    Parameters
    ----------
    anchor_t : np.ndarray, shape (T, 3)
    out_t    : np.ndarray, shape (T, 3)
    bins     : int

    Returns
    -------
    float
        L1 distance between normalized 3D occupancy histograms.
    """
    anchor = np.asarray(anchor_t, dtype=np.float64)
    pred = np.asarray(out_t, dtype=np.float64)

    if anchor.ndim != 2 or anchor.shape[1] != 3:
        raise ValueError(f'anchor_t must have shape (T,3), got {anchor.shape}')
    if pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError(f'out_t must have shape (T,3), got {pred.shape}')

    mins = np.minimum(anchor.min(axis=0), pred.min(axis=0))
    maxs = np.maximum(anchor.max(axis=0), pred.max(axis=0))
    ranges = [(mins[i], maxs[i]) for i in range(3)]

    hist_true, edges = np.histogramdd(anchor, bins=bins, range=ranges)
    hist_pred, _ = np.histogramdd(pred, bins=edges)

    hist_true = hist_true.reshape(-1)
    hist_pred = hist_pred.reshape(-1)

    denom_true = hist_true.sum() + 1e-12
    denom_pred = hist_pred.sum() + 1e-12

    hist_true = hist_true / denom_true
    hist_pred = hist_pred / denom_pred

    return float(np.abs(hist_true - hist_pred).sum())


def _safe_standardize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + eps)


def _acf_np(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x - x.mean()
    denom = np.dot(x, x) + 1e-12
    max_lag = int(max_lag)
    max_lag = max(1, min(max_lag, x.shape[0] - 1))
    acf = []
    for lag in range(1, max_lag + 1):
        acf.append(np.dot(x[:-lag], x[lag:]) / denom)
    return np.asarray(acf, dtype=np.float64)


def _psd_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    fft_vals = np.fft.rfft(x, axis=0)
    power = (fft_vals.real ** 2 + fft_vals.imag ** 2) / max(1, x.shape[0])
    return power


def _to_prob_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, 0.0)
    s = float(np.sum(x))
    if s <= eps:
        return np.full_like(x, 1.0 / max(1, x.size), dtype=np.float64)
    return x / s


def _js_divergence_np(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Jensen-Shannon divergence between discrete distributions.
    Returns a value in [0, ln(2)].
    """
    p = _to_prob_np(p, eps=eps)
    q = _to_prob_np(q, eps=eps)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * (np.log(a + eps) - np.log(b + eps))))

    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def cal_l63_attractor_metrics(
    anchor_t: np.ndarray,
    out_t: np.ndarray,
    max_lag: int = 64,
) -> dict[str, float]:
    """
    Evaluate attractor-sensitive diagnostics to detect limit-cycle collapse.
    """
    anchor = np.asarray(anchor_t, dtype=np.float64)
    pred = np.asarray(out_t, dtype=np.float64)

    if anchor.ndim != 2 or anchor.shape[1] != 3:
        raise ValueError(f'anchor_t must have shape (T,3), got {anchor.shape}')
    if pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError(f'out_t must have shape (T,3), got {pred.shape}')

    t_len = min(anchor.shape[0], pred.shape[0])
    anchor = anchor[:t_len]
    pred = pred[:t_len]

    anchor_s = _safe_standardize_np(anchor)
    pred_s = _safe_standardize_np(pred)

    psd_anchor = _psd_np(anchor_s)
    psd_pred = _psd_np(pred_s)
    psd_log_l1 = float(np.mean(np.abs(np.log(psd_anchor + 1e-8) - np.log(psd_pred + 1e-8))))
    # Spectral-shape mismatch on normalized PSDs, averaged over x/y/z.
    # We drop DC since it is removed by centering and can be numerically noisy.
    if psd_anchor.shape[0] > 1 and psd_pred.shape[0] > 1:
        psd_anchor_nz = psd_anchor[1:, :]
        psd_pred_nz = psd_pred[1:, :]
    else:
        psd_anchor_nz = psd_anchor
        psd_pred_nz = psd_pred
    js_vals = [
        _js_divergence_np(psd_anchor_nz[:, d], psd_pred_nz[:, d]) / np.log(2.0)
        for d in range(psd_anchor_nz.shape[1])
    ]
    psd_jsd = float(np.mean(js_vals))

    acf_anchor_x = _acf_np(anchor_s[:, 0], max_lag=max_lag)
    acf_pred_x = _acf_np(pred_s[:, 0], max_lag=max_lag)
    acf_anchor_z = _acf_np(anchor_s[:, 2], max_lag=max_lag)
    acf_pred_z = _acf_np(pred_s[:, 2], max_lag=max_lag)
    autocorr_l1 = float(
        0.5 * np.mean(np.abs(acf_anchor_x - acf_pred_x))
        + 0.5 * np.mean(np.abs(acf_anchor_z - acf_pred_z))
    )

    sign_anchor = np.sign(anchor[:, 0])
    sign_pred = np.sign(pred[:, 0])

    occ_anchor = float(np.mean(sign_anchor > 0))
    occ_pred = float(np.mean(sign_pred > 0))
    regime_occupancy_abs = float(abs(occ_anchor - occ_pred))

    def _switch_rate(sign_series: np.ndarray) -> float:
        if sign_series.shape[0] < 2:
            return 0.0
        valid = sign_series.copy()
        valid[valid == 0] = 1
        return float(np.mean(valid[1:] != valid[:-1]))

    switch_anchor = _switch_rate(sign_anchor)
    switch_pred = _switch_rate(sign_pred)
    switch_rate_abs = float(abs(switch_anchor - switch_pred))

    return {
        'psd_log_l1': psd_log_l1,
        'psd_jsd': psd_jsd,
        'autocorr_l1': autocorr_l1,
        'regime_occupancy_abs': regime_occupancy_abs,
        'switch_rate_abs': switch_rate_abs,
    }


def _safe_standardize_torch(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True)
    return (x - mean) / (std + eps)


def _acf_torch_mean_abs_diff(x_true: torch.Tensor, x_pred: torch.Tensor, max_lag: int) -> torch.Tensor:
    """
    x_true/x_pred: (B, T)
    """
    max_lag = max(1, min(int(max_lag), x_true.shape[1] - 1))
    true_center = x_true - x_true.mean(dim=1, keepdim=True)
    pred_center = x_pred - x_pred.mean(dim=1, keepdim=True)

    denom_true = (true_center * true_center).sum(dim=1, keepdim=True) + 1e-6
    denom_pred = (pred_center * pred_center).sum(dim=1, keepdim=True) + 1e-6

    diffs = []
    for lag in range(1, max_lag + 1):
        acf_true = (true_center[:, :-lag] * true_center[:, lag:]).sum(dim=1, keepdim=True) / denom_true
        acf_pred = (pred_center[:, :-lag] * pred_center[:, lag:]).sum(dim=1, keepdim=True) / denom_pred
        diffs.append(torch.abs(acf_true - acf_pred))
    return torch.stack(diffs, dim=0).mean()


def cal_l63_attractor_penalty_torch(
    anchor_t: torch.Tensor,
    out_t: torch.Tensor,
    *,
    max_lag: int = 64,
    psd_weight: float = 1.0,
    autocorr_weight: float = 1.0,
    regime_weight: float = 0.5,
    switch_weight: float = 0.5,
    regime_sharpness: float = 8.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Differentiable attractor-aware penalty for L63 training.
    """
    _validate_l63_batch(anchor_t, out_t)

    t_len = min(anchor_t.shape[1], out_t.shape[1])
    anchor = anchor_t[:, :t_len, :]
    pred = out_t[:, :t_len, :]
    anchor_s = _safe_standardize_torch(anchor)
    pred_s = _safe_standardize_torch(pred)

    fft_true = torch.fft.rfft(anchor_s, dim=1)
    fft_pred = torch.fft.rfft(pred_s, dim=1)
    psd_true = (fft_true.real ** 2 + fft_true.imag ** 2) / max(1, t_len)
    psd_pred = (fft_pred.real ** 2 + fft_pred.imag ** 2) / max(1, t_len)
    loss_psd = torch.mean(torch.abs(torch.log(psd_true + 1e-6) - torch.log(psd_pred + 1e-6)))

    loss_acf_x = _acf_torch_mean_abs_diff(anchor_s[:, :, 0], pred_s[:, :, 0], max_lag=max_lag)
    loss_acf_z = _acf_torch_mean_abs_diff(anchor_s[:, :, 2], pred_s[:, :, 2], max_lag=max_lag)
    loss_autocorr = 0.5 * (loss_acf_x + loss_acf_z)

    soft_true = torch.tanh(regime_sharpness * anchor[:, :, 0])
    soft_pred = torch.tanh(regime_sharpness * pred[:, :, 0])
    occ_true = ((soft_true + 1.0) * 0.5).mean(dim=1)
    occ_pred = ((soft_pred + 1.0) * 0.5).mean(dim=1)
    loss_regime = torch.mean(torch.abs(occ_true - occ_pred))

    if t_len > 1:
        sw_true = 0.5 * torch.abs(soft_true[:, 1:] - soft_true[:, :-1]).mean(dim=1)
        sw_pred = 0.5 * torch.abs(soft_pred[:, 1:] - soft_pred[:, :-1]).mean(dim=1)
        loss_switch = torch.mean(torch.abs(sw_true - sw_pred))
    else:
        loss_switch = torch.zeros((), device=anchor_t.device, dtype=anchor_t.dtype)

    total = (
        float(psd_weight) * loss_psd
        + float(autocorr_weight) * loss_autocorr
        + float(regime_weight) * loss_regime
        + float(switch_weight) * loss_switch
    )
    components = {
        'psd': loss_psd,
        'autocorr': loss_autocorr,
        'regime_occ': loss_regime,
        'switch_rate': loss_switch,
    }
    return total, components
