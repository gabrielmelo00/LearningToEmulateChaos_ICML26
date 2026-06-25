"""
Lorenz-63 utilities for dataset generation.

This module provides:
- deterministic RK4 integration for Lorenz-63,
- parameter sampling around the canonical chaotic setting,
- trajectory segment generation with burn-in and subsampling.
"""

from __future__ import annotations

import numpy as np


DEFAULT_SIGMA = 10.0
DEFAULT_RHO = 28.0
DEFAULT_BETA = 8.0 / 3.0


def lorenz63_rhs(
    state: np.ndarray,
    sigma: float,
    rho: float,
    beta: float,
) -> np.ndarray:
    """Lorenz-63 vector field."""
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz], dtype=np.float64)


def rk4_step(
    state: np.ndarray,
    dt: float,
    sigma: float,
    rho: float,
    beta: float,
) -> np.ndarray:
    """Single RK4 step."""
    k1 = lorenz63_rhs(state, sigma, rho, beta)
    k2 = lorenz63_rhs(state + 0.5 * dt * k1, sigma, rho, beta)
    k3 = lorenz63_rhs(state + 0.5 * dt * k2, sigma, rho, beta)
    k4 = lorenz63_rhs(state + dt * k3, sigma, rho, beta)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def sample_l63_params(
    seed: int,
    sigma_center: float = DEFAULT_SIGMA,
    rho_center: float = DEFAULT_RHO,
    beta_center: float = DEFAULT_BETA,
    sigma_delta: float = 1.0,
    rho_delta: float = 2.0,
    beta_delta: float = 0.2,
) -> np.ndarray:
    """
    Sample (sigma, rho, beta) uniformly around canonical chaotic values.
    """
    rng = np.random.default_rng(seed)
    sigma = rng.uniform(sigma_center - sigma_delta, sigma_center + sigma_delta)
    rho = rng.uniform(rho_center - rho_delta, rho_center + rho_delta)
    beta = rng.uniform(beta_center - beta_delta, beta_center + beta_delta)
    return np.array([sigma, rho, beta], dtype=np.float32)


def sample_l63_ic(seed: int, scale: float = 8.0) -> np.ndarray:
    """
    Sample an initial condition for Lorenz-63 integration.

    The z component is shifted upward so trajectories quickly approach the
    butterfly attractor lobes after burn-in.
    """
    rng = np.random.default_rng(seed)
    ic = rng.normal(size=3).astype(np.float64) * scale
    ic[2] += 20.0
    return ic.astype(np.float32)


def generate_l63_segment(
    params: np.ndarray,
    initial_state: np.ndarray,
    dt: float = 0.01,
    n_steps: int = 10_000,
    stride: int = 10,
    burnin_steps: int = 5_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate a Lorenz-63 segment.

    Parameters
    ----------
    params : array-like, shape (3,)
        [sigma, rho, beta]
    initial_state : array-like, shape (3,)
    dt : float
        Integrator step size.
    n_steps : int
        Number of post-burn-in integration steps.
    stride : int
        Keep every `stride`-th state.
    burnin_steps : int
        Number of initial steps discarded before recording.

    Returns
    -------
    traj : np.ndarray, shape (n_steps // stride, 3)
    final_state : np.ndarray, shape (3,)
    """
    sigma, rho, beta = [float(v) for v in params]
    state = np.asarray(initial_state, dtype=np.float64).copy()

    for _ in range(int(burnin_steps)):
        state = rk4_step(state, dt, sigma, rho, beta)

    frames: list[np.ndarray] = []
    for step in range(int(n_steps)):
        if step % int(stride) == 0:
            frames.append(state.astype(np.float32).copy())
        state = rk4_step(state, dt, sigma, rho, beta)

    traj = np.asarray(frames, dtype=np.float32)
    return traj, state.astype(np.float32)
