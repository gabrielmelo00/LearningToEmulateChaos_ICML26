import numpy as np


class ETDRK4:
    """
    Exponential Time Differencing RK4 for diagonal linear operators.
    """

    def __init__(self, lin, dt, M=32):
        self.dt = dt
        c = lin.astype(complex) * dt
        c2 = c / 2

        self.E = np.exp(c)
        self.E2 = np.exp(c2)

        r = np.exp(2j * np.pi * (np.arange(1, M + 1) - 0.5) / M)

        Z2 = c2[:, None] + r[None, :]
        eZ2 = np.exp(Z2)
        phi1_h = np.real(np.mean((eZ2 - 1) / Z2, axis=1))
        self.Q = (dt / 2) * phi1_h

        Z = c[:, None] + r[None, :]
        eZ = np.exp(Z)
        phi1 = np.real(np.mean((eZ - 1) / Z, axis=1))
        phi2 = np.real(np.mean((eZ - 1 - Z) / Z**2, axis=1))
        phi3 = np.real(np.mean((eZ - 1 - Z - Z**2 / 2) / Z**3, axis=1))

        self.f1 = dt * (phi1 - 3 * phi2 + 4 * phi3)
        self.f2 = dt * (2 * phi2 - 4 * phi3)
        self.f3 = dt * (-phi2 + 4 * phi3)

    def step(self, u_hat, N_fn):
        Nu = N_fn(u_hat)
        a = self.E2 * u_hat + self.Q * Nu
        Na = N_fn(a)
        b = self.E2 * u_hat + self.Q * Na
        Nb = N_fn(b)
        c = self.E2 * a + self.Q * (2 * Nb - Nu)
        Nc = N_fn(c)
        return self.E * u_hat + self.f1 * Nu + self.f2 * (Na + Nb) + self.f3 * Nc


class KS:
    """
    Spectral solver for the Kuramoto-Sivashinsky equation:
        u_t + u u_x + u_xx + u_xxxx = 0
    on [0, 2*pi*L] with periodic boundary conditions.
    """

    def __init__(self, L=50, N=256, dt=0.1, initial_condition=None):
        self.L = L
        self.n = N
        self.dt = dt

        kk = N * np.fft.fftfreq(N)[: N // 2 + 1]
        self.k = kk.astype(np.float64) / L
        self.ik = 1j * self.k.copy()
        self.ik[-1] = 0.0

        lin = self.k**2 - self.k**4
        self.stepper = ETDRK4(lin, dt)

        if initial_condition is None:
            x = np.cos(np.pi * np.linspace(0, 2 * np.pi * L, N, endpoint=False) / L)
        else:
            x = np.asarray(initial_condition, dtype=np.float64)

        self.x = x - x.mean()
        self.xspec = np.fft.rfft(self.x)

    def _nlterm(self, xspec):
        xspec_da = xspec.copy()
        xspec_da[2 * len(xspec) // 3:] = 0
        u = np.fft.irfft(xspec_da)
        return -0.5 * self.ik * np.fft.rfft(u**2)

    def advance(self):
        self.xspec = self.stepper.step(self.xspec, self._nlterm)
        self.x = np.fft.irfft(self.xspec)

    def run(self, n_steps):
        for _ in range(n_steps):
            self.advance()

    def snapshot(self):
        return self.x.copy()


def sample_ic(seed, N=256, L=50, n_modes=4):
    np.random.seed(seed)
    x = np.linspace(0, 2 * np.pi * L, N, endpoint=False)
    ic = np.zeros(N)
    for k in range(1, n_modes + 1):
        amp = np.random.randn()
        phi = np.random.uniform(0, 2 * np.pi)
        ic += amp * np.cos(k * x / L + phi)
    return (ic - ic.mean()).astype(np.float32)


def generate_ks_segment(initial_condition, L=50, N=256, dt=0.1, T=200, t_res=10):
    """
    Integrate one contiguous KS segment.

    Returns
    -------
    traj : np.ndarray, shape (T / (dt*t_res), N)
    final_state : np.ndarray, shape (N,)
    """
    solver = KS(L=L, N=N, dt=dt, initial_condition=initial_condition)
    n_steps = int(T / dt)
    frames = []
    for step in range(n_steps):
        if step % t_res == 0:
            frames.append(solver.snapshot())
        solver.advance()
    return np.asarray(frames, dtype=np.float32), solver.snapshot().astype(np.float32)
