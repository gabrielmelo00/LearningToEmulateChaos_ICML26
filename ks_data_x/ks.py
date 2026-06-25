"""
ks.py
-----
KS spectral solver and single-trajectory data generation function.
Equivalent to l96.py in the L96 pipeline.

Key differences from L96:
- No physical parameter F. The operator input is the initial condition u0.
- State is a spatial field u(x) of dimension N (grid points), not N scalar variables.
- ETDRK4 timestepper in Fourier space (no scipy solve_ivp needed).
- Burn-in is applied inside generate_ks_data to ensure the IC is on the attractor.
"""

import numpy as np
import torch


# ─────────────────────────────────────────────
# ETDRK4 (self-contained, no external deps)
# ─────────────────────────────────────────────

class ETDRK4:
    """
    Exponential Time Differencing RK4 (Cox & Matthews 2002,
    stabilised via Kassam & Trefethen 2005 contour integrals).

    Solves:  du/dt = L*u + N(u)
    L is a diagonal linear operator in Fourier space.
    All exponential factors are precomputed at construction time.
    """

    def __init__(self, lin, dt, M=32):
        self.dt = dt
        c  = lin.astype(complex) * dt
        c2 = c / 2

        self.E  = np.exp(c)
        self.E2 = np.exp(c2)

        r = np.exp(2j * np.pi * (np.arange(1, M + 1) - 0.5) / M)

        Z2  = c2[:, None] + r[None, :]
        eZ2 = np.exp(Z2)
        phi1_h = np.real(np.mean((eZ2 - 1) / Z2,         axis=1))
        self.Q  = (dt / 2) * phi1_h

        Z  = c[:, None] + r[None, :]
        eZ = np.exp(Z)
        phi1 = np.real(np.mean((eZ - 1) / Z,                        axis=1))
        phi2 = np.real(np.mean((eZ - 1 - Z) / Z**2,                 axis=1))
        phi3 = np.real(np.mean((eZ - 1 - Z - Z**2/2) / Z**3,        axis=1))

        self.f1 = dt * (phi1 - 3*phi2 + 4*phi3)
        self.f2 = dt * (       2*phi2 - 4*phi3)
        self.f3 = dt * (      -  phi2 + 4*phi3)

    def step(self, u_hat, N_fn):
        Nu  = N_fn(u_hat)
        a   = self.E2 * u_hat + self.Q * Nu
        Na  = N_fn(a)
        b   = self.E2 * u_hat + self.Q * Na
        Nb  = N_fn(b)
        c   = self.E2 * a    + self.Q * (2*Nb - Nu)
        Nc  = N_fn(c)
        return self.E * u_hat + self.f1*Nu + self.f2*(Na + Nb) + self.f3*Nc


# ─────────────────────────────────────────────
# KS solver class
# ─────────────────────────────────────────────

class KS:
    """
    Spectral solver for the Kuramoto-Sivashinsky equation:

        u_t + u*u_x + u_xx + u_xxxx = 0   on [0, 2*pi*L], periodic BCs

    All physical coefficients are fixed (standard KS).
    Only the initial condition varies across trajectories.
    """

    def __init__(self, L=50, N=256, dt=0.25, initial_condition=None):
        self.L = L
        self.n = N
        self.dt = dt

        kk      = N * np.fft.fftfreq(N)[: N // 2 + 1]
        self.k  = kk.astype(np.float64) / L
        self.ik = 1j * self.k.copy()
        self.ik[-1] = 0.0                          # zero Nyquist mode

        lin = self.k**2 - self.k**4               # instability - dissipation
        self.stepper = ETDRK4(lin, dt)

        if initial_condition is None:
            x = np.cos(np.pi * np.linspace(0, 2*np.pi*L, N, endpoint=False) / L)
        else:
            x = np.asarray(initial_condition, dtype=np.float64)

        self.x     = x - x.mean()
        self.xspec = np.fft.rfft(self.x)

    def get_domain(self):
        return np.linspace(0, 2*np.pi*self.L, self.n, endpoint=False)

    def _nlterm(self, xspec):
        xspec_da = xspec.copy()
        xspec_da[2 * len(xspec) // 3:] = 0        # 2/3 dealiasing
        u = np.fft.irfft(xspec_da)
        return -0.5 * self.ik * np.fft.rfft(u**2)

    def advance(self):
        self.xspec = self.stepper.step(self.xspec, self._nlterm)
        self.x     = np.fft.irfft(self.xspec)

    def run(self, n_steps):
        for _ in range(n_steps):
            self.advance()

    def snapshot(self):
        return self.x.copy()


# ─────────────────────────────────────────────
# Data generation function
# ─────────────────────────────────────────────

def generate_ks_data(args,
                     L=50, N=256, dt=0.25,
                     T_burnin=500.0,
                     T_total=500.0,
                     t_res=1):
    """
    Generate a single KS trajectory (after burn-in) for dataset construction.
    Equivalent to generate_l96_data in the L96 pipeline.

    Parameters
    ----------
    args : array-like, shape (N+1,)
        Concatenation of [initial_condition (N,), seed (1,)].
        The IC is used only for the burn-in kick; after burn-in the solver
        is on the attractor regardless of the starting point.
    L : float
        Domain length parameter. L=50 is the standard chaotic setting.
    N : int
        Number of spatial grid points (state dimension).
    dt : float
        Solver time step.
    T_burnin : float
        Time units to discard before recording (ensures attractor sampling).
    T_total : float
        Time units to record after burn-in.
    t_res : int
        Store every t_res-th time step (subsampling factor).

    Returns
    -------
    np.ndarray, shape (T_total // (dt * t_res), N)
        Trajectory on the attractor, subsampled at t_res.
    """
    seed = int(args[-1])
    ic   = args[:-1]                               # shape (N,)
    np.random.seed(seed)

    solver = KS(L=L, N=N, dt=dt, initial_condition=ic)

    solver.run(int(T_burnin / dt))

    n_total = int(T_total / dt)
    frames  = []
    for step in range(n_total):
        if step % t_res == 0:
            frames.append(solver.snapshot())
        solver.advance()

    return np.array(frames, dtype=np.float32)      # (T_stored, N)