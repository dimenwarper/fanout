#!/usr/bin/env python3
"""Generate high-accuracy reference solutions for PDE benchmark tasks.

Methods:
- Burgers 1D: Pseudo-spectral (FFT) at 512 points, downsampled to 128
- Navier-Stokes 2D: Pseudo-spectral (Fourier-Galerkin) at 256x256, downsampled to 64x64
- KS 1D: ETDRK4 at 1024 points, downsampled to 256

Each task generates 20 test instances with different initial conditions.
Reference solutions are trajectory snapshots (10 time points per instance).
Files: {task}_ic_{idx:02d}.npy, {task}_ref_{idx:02d}.npy, {task}_t_coordinates.npy

Usage:
    python benchmarks/pde-solvers/reference/generate_references.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REF_DIR = Path(__file__).resolve().parent

N_INSTANCES = 20
N_SNAPSHOTS = 10


# -- Initial Conditions -------------------------------------------------------


def burgers_ics(nx: int, n: int = N_INSTANCES) -> list[np.ndarray]:
    """Generate n initial conditions for 1D Burgers on [0, 2pi].

    First 3 are the original hand-crafted ICs; remainder are random
    superpositions of low Fourier modes (wavenumbers 1-5).
    """
    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    ics: list[np.ndarray] = [
        np.sin(x),
        np.sin(x) + 0.5 * np.sin(2 * x),
        np.exp(-10 * (x - np.pi) ** 2),
    ]
    rng = np.random.default_rng(seed=42)
    while len(ics) < n:
        u = np.zeros(nx)
        n_modes = rng.integers(2, 6)
        for _ in range(n_modes):
            wn = rng.integers(1, 6)
            amp = rng.uniform(0.3, 1.5)
            phase = rng.uniform(0, 2 * np.pi)
            u += amp * np.sin(wn * x + phase)
        ics.append(u)
    return ics


def ns_ics(nx: int, n: int = N_INSTANCES) -> list[np.ndarray]:
    """Generate n initial vorticity fields for 2D NS on [0, 2pi]^2.

    First 2 are hand-crafted; remainder are random 2D Fourier mode superpositions.
    """
    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    ics: list[np.ndarray] = [
        2 * np.cos(X) * np.cos(Y),
        np.sin(2 * X) * np.cos(2 * Y) + 0.5 * np.cos(3 * X) * np.sin(3 * Y),
    ]
    rng = np.random.default_rng(seed=42)
    while len(ics) < n:
        omega = np.zeros((nx, nx))
        n_modes = rng.integers(3, 7)
        for _ in range(n_modes):
            kx = rng.integers(1, 5)
            ky = rng.integers(1, 5)
            amp = rng.uniform(0.3, 1.5)
            phase_x = rng.uniform(0, 2 * np.pi)
            phase_y = rng.uniform(0, 2 * np.pi)
            omega += amp * np.sin(kx * X + phase_x) * np.cos(ky * Y + phase_y)
        ics.append(omega)
    return ics


def ks_ics(nx: int, n: int = N_INSTANCES) -> list[np.ndarray]:
    """Generate n initial conditions for 1D KS on [0, 64pi].

    First 3 are hand-crafted; remainder are random low-mode superpositions.
    """
    L = 64 * np.pi
    x = np.linspace(0, L, nx, endpoint=False)
    ics: list[np.ndarray] = [
        np.cos(x / 16) * (1 + np.sin(x / 16)),
        np.sin(2 * np.pi * x / L) + 0.5 * np.sin(4 * np.pi * x / L),
        np.exp(-0.01 * (x - L / 2) ** 2) * np.cos(4 * np.pi * x / L),
    ]
    rng = np.random.default_rng(seed=42)
    while len(ics) < n:
        u = np.zeros(nx)
        n_modes = rng.integers(2, 6)
        for _ in range(n_modes):
            wn = rng.integers(1, 6)
            amp = rng.uniform(0.3, 1.5)
            phase = rng.uniform(0, 2 * np.pi)
            u += amp * np.sin(2 * np.pi * wn * x / L + phase)
        ics.append(u)
    return ics


# -- Solvers (trajectory versions) --------------------------------------------


def solve_burgers_spectral(ic: np.ndarray, nx: int, t_coords: np.ndarray, nu: float) -> np.ndarray:
    """Pseudo-spectral solver for 1D Burgers with RK4, returning trajectory snapshots.

    Returns array of shape (len(t_coords), nx).
    """
    dx = 2 * np.pi / nx
    k = np.fft.fftfreq(nx, d=dx / (2 * np.pi))
    k2 = k**2

    dt = min(0.5 * dx / (np.max(np.abs(ic)) + 1e-10), 0.5 / (nu * np.max(k2) + 1e-10))
    dt = min(dt, 0.001)

    def rhs(u_hat):
        u = np.fft.ifft(u_hat).real
        nonlinear = -0.5 * 1j * k * np.fft.fft(u**2)
        diffusion = -nu * k2 * u_hat
        return nonlinear + diffusion

    u_hat = np.fft.fft(ic)
    t = 0.0
    snapshots = []
    t_idx = 0

    while t_idx < len(t_coords):
        target = t_coords[t_idx]
        while t < target - 1e-14:
            step = min(dt, target - t)
            k1 = step * rhs(u_hat)
            k2_ = step * rhs(u_hat + 0.5 * k1)
            k3 = step * rhs(u_hat + 0.5 * k2_)
            k4 = step * rhs(u_hat + k3)
            u_hat = u_hat + (k1 + 2 * k2_ + 2 * k3 + k4) / 6
            t += step
        snapshots.append(np.fft.ifft(u_hat).real.copy())
        t_idx += 1

    return np.array(snapshots)


def solve_ns_spectral(ic: np.ndarray, nx: int, t_coords: np.ndarray, nu: float) -> np.ndarray:
    """Pseudo-spectral solver for 2D NS vorticity with RK4, returning trajectory snapshots.

    Returns array of shape (len(t_coords), nx, nx).
    """
    dx = 2 * np.pi / nx
    kx = np.fft.fftfreq(nx, d=dx / (2 * np.pi))
    ky = np.fft.fftfreq(nx, d=dx / (2 * np.pi))
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX**2 + KY**2
    K2_safe = K2.copy()
    K2_safe[0, 0] = 1.0

    kmax = nx // 3
    dealias = np.ones((nx, nx))
    dealias[np.abs(KX) > kmax] = 0.0
    dealias[np.abs(KY) > kmax] = 0.0

    dt = min(0.5 * dx / (np.max(np.abs(ic)) + 1e-10), 0.25 * dx**2 / nu)
    dt = min(dt, 0.002)

    def rhs(omega_hat):
        psi_hat = omega_hat / K2_safe
        psi_hat[0, 0] = 0.0
        u_hat = 1j * KY * psi_hat
        v_hat = -1j * KX * psi_hat
        domega_dx_hat = 1j * KX * omega_hat
        domega_dy_hat = 1j * KY * omega_hat
        u = np.fft.ifft2(u_hat).real
        v = np.fft.ifft2(v_hat).real
        domega_dx = np.fft.ifft2(domega_dx_hat).real
        domega_dy = np.fft.ifft2(domega_dy_hat).real
        advection = np.fft.fft2(u * domega_dx + v * domega_dy) * dealias
        diffusion = -nu * K2 * omega_hat
        return -advection + diffusion

    omega_hat = np.fft.fft2(ic) * dealias
    t = 0.0
    snapshots = []
    t_idx = 0

    while t_idx < len(t_coords):
        target = t_coords[t_idx]
        while t < target - 1e-14:
            step = min(dt, target - t)
            k1 = step * rhs(omega_hat)
            k2_ = step * rhs(omega_hat + 0.5 * k1)
            k3 = step * rhs(omega_hat + 0.5 * k2_)
            k4 = step * rhs(omega_hat + k3)
            omega_hat = (omega_hat + (k1 + 2 * k2_ + 2 * k3 + k4) / 6) * dealias
            t += step
        snapshots.append(np.fft.ifft2(omega_hat).real.copy())
        t_idx += 1

    return np.array(snapshots)


def solve_ks_etdrk4(ic: np.ndarray, nx: int, t_coords: np.ndarray) -> np.ndarray:
    """ETDRK4 solver for 1D Kuramoto-Sivashinsky, returning trajectory snapshots.

    Returns array of shape (len(t_coords), nx).
    """
    L = 64 * np.pi
    k = 2 * np.pi / L * np.fft.fftfreq(nx) * nx
    Lk = k**2 - k**4

    dt = 0.5
    # We'll step through and record snapshots at target times
    # Use fixed dt but adjust final step to land on target times

    # ETDRK4 coefficients (Kassam & Trefethen, 2005)
    E = np.exp(Lk * dt)
    E2 = np.exp(Lk * dt / 2)

    M = 32
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * Lk[:, None] + r[None, :]

    Q = dt * np.real(np.mean(((np.exp(LR / 2) - 1) / LR), axis=1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1))
    f3 = dt * np.real(np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1))

    # Dealiasing mask (2/3 rule)
    dealias = np.ones(nx)
    dealias[nx // 3 : 2 * nx // 3 + 1] = 0.0

    def nonlinear(u_hat):
        u = np.fft.ifft(u_hat * dealias).real
        return -0.5 * 1j * k * np.fft.fft(u**2) * dealias

    def etdrk4_step(u_hat, E, E2, Q, f1, f2, f3):
        Nu = nonlinear(u_hat)
        a = E2 * u_hat + Q * Nu
        Na = nonlinear(a)
        b = E2 * u_hat + Q * Na
        Nb = nonlinear(b)
        c = E2 * a + Q * (2 * Nb - Nu)
        Nc = nonlinear(c)
        return E * u_hat + Nu * f1 + 2 * (Na + Nb) * f2 + Nc * f3

    def compute_coeffs(dt_local):
        """Compute ETDRK4 coefficients for a given dt."""
        E_l = np.exp(Lk * dt_local)
        E2_l = np.exp(Lk * dt_local / 2)
        LR_l = dt_local * Lk[:, None] + r[None, :]
        Q_l = dt_local * np.real(np.mean(((np.exp(LR_l / 2) - 1) / LR_l), axis=1))
        f1_l = dt_local * np.real(np.mean((-4 - LR_l + np.exp(LR_l) * (4 - 3 * LR_l + LR_l**2)) / LR_l**3, axis=1))
        f2_l = dt_local * np.real(np.mean((2 + LR_l + np.exp(LR_l) * (-2 + LR_l)) / LR_l**3, axis=1))
        f3_l = dt_local * np.real(np.mean((-4 - 3 * LR_l - LR_l**2 + np.exp(LR_l) * (4 - LR_l)) / LR_l**3, axis=1))
        return E_l, E2_l, Q_l, f1_l, f2_l, f3_l

    u_hat = np.fft.fft(ic)
    t = 0.0
    snapshots = []
    t_idx = 0

    while t_idx < len(t_coords):
        target = t_coords[t_idx]
        while t < target - 1e-14:
            remaining = target - t
            if remaining < dt - 1e-14:
                # Need a smaller step — recompute coefficients
                E_s, E2_s, Q_s, f1_s, f2_s, f3_s = compute_coeffs(remaining)
                u_hat = etdrk4_step(u_hat, E_s, E2_s, Q_s, f1_s, f2_s, f3_s)
                t += remaining
            else:
                u_hat = etdrk4_step(u_hat, E, E2, Q, f1, f2, f3)
                t += dt
        snapshots.append(np.fft.ifft(u_hat).real.copy())
        t_idx += 1

    return np.array(snapshots)


# -- Downsampling --------------------------------------------------------------


def downsample_1d(u: np.ndarray, target_nx: int) -> np.ndarray:
    """Downsample 1D array via spectral truncation."""
    u_hat = np.fft.fft(u)
    n = len(u)
    nt = target_nx
    u_hat_trunc = np.zeros(nt, dtype=complex)
    half = nt // 2
    u_hat_trunc[:half] = u_hat[:half]
    u_hat_trunc[-half:] = u_hat[-half:]
    return np.fft.ifft(u_hat_trunc).real * (nt / n)


def downsample_2d(u: np.ndarray, target_nx: int) -> np.ndarray:
    """Downsample 2D array via spectral truncation."""
    n = u.shape[0]
    nt = target_nx
    u_hat = np.fft.fft2(u)
    u_hat_trunc = np.zeros((nt, nt), dtype=complex)
    half = nt // 2
    u_hat_trunc[:half, :half] = u_hat[:half, :half]
    u_hat_trunc[:half, -half:] = u_hat[:half, -half:]
    u_hat_trunc[-half:, :half] = u_hat[-half:, :half]
    u_hat_trunc[-half:, -half:] = u_hat[-half:, -half:]
    return np.fft.ifft2(u_hat_trunc).real * (nt / n) ** 2


# -- Main ----------------------------------------------------------------------


def generate_burgers():
    """Generate Burgers reference solutions at 512 points, downsample to 128."""
    print("Generating Burgers 1D references...")
    nx_hi = 512
    nx_lo = 128
    nu = 0.01
    t_final = 5.0

    t_coords = np.linspace(t_final / N_SNAPSHOTS, t_final, N_SNAPSHOTS)
    np.save(REF_DIR / "burgers_1d_t_coordinates.npy", t_coords)

    ics = burgers_ics(nx_hi)
    for idx, ic in enumerate(ics):
        print(f"  Instance {idx:02d}/{len(ics)}...", end=" ", flush=True)
        # Solve at high resolution, get trajectory
        traj_hi = solve_burgers_spectral(ic, nx_hi, t_coords, nu)
        # Downsample each snapshot
        traj_lo = np.array([downsample_1d(snap, nx_lo) for snap in traj_hi])

        # IC at low resolution
        ic_lo = downsample_1d(ic, nx_lo)
        np.save(REF_DIR / f"burgers_1d_ic_{idx:02d}.npy", ic_lo)
        np.save(REF_DIR / f"burgers_1d_ref_{idx:02d}.npy", traj_lo)
        print(f"done (traj shape={traj_lo.shape}, max={np.max(np.abs(traj_lo)):.4f})")

    print(f"  Saved {len(ics)} Burgers references.\n")


def generate_navier_stokes():
    """Generate NS reference solutions at 256x256, downsample to 64x64."""
    print("Generating Navier-Stokes 2D references...")
    nx_hi = 256
    nx_lo = 64
    nu = 1e-3
    t_final = 10.0

    t_coords = np.linspace(t_final / N_SNAPSHOTS, t_final, N_SNAPSHOTS)
    np.save(REF_DIR / "navier_stokes_2d_t_coordinates.npy", t_coords)

    ics = ns_ics(nx_hi)
    for idx, ic in enumerate(ics):
        print(f"  Instance {idx:02d}/{len(ics)}...", end=" ", flush=True)
        traj_hi = solve_ns_spectral(ic, nx_hi, t_coords, nu)
        traj_lo = np.array([downsample_2d(snap, nx_lo) for snap in traj_hi])

        ic_lo = downsample_2d(ic, nx_lo)
        np.save(REF_DIR / f"navier_stokes_2d_ic_{idx:02d}.npy", ic_lo)
        np.save(REF_DIR / f"navier_stokes_2d_ref_{idx:02d}.npy", traj_lo)
        print(f"done (traj shape={traj_lo.shape}, max={np.max(np.abs(traj_lo)):.4f})")

    print(f"  Saved {len(ics)} Navier-Stokes references.\n")


def generate_ks():
    """Generate KS reference solutions at 1024 points, downsample to 256."""
    print("Generating Kuramoto-Sivashinsky 1D references...")
    nx_hi = 1024
    nx_lo = 256
    t_final = 200.0

    t_coords = np.linspace(t_final / N_SNAPSHOTS, t_final, N_SNAPSHOTS)
    np.save(REF_DIR / "ks_1d_t_coordinates.npy", t_coords)

    ics = ks_ics(nx_hi)
    for idx, ic in enumerate(ics):
        print(f"  Instance {idx:02d}/{len(ics)}...", end=" ", flush=True)
        traj_hi = solve_ks_etdrk4(ic, nx_hi, t_coords)
        traj_lo = np.array([downsample_1d(snap, nx_lo) for snap in traj_hi])

        ic_lo = downsample_1d(ic, nx_lo)
        np.save(REF_DIR / f"ks_1d_ic_{idx:02d}.npy", ic_lo)
        np.save(REF_DIR / f"ks_1d_ref_{idx:02d}.npy", traj_lo)
        print(f"done (traj shape={traj_lo.shape}, max={np.max(np.abs(traj_lo)):.4f})")

    print(f"  Saved {len(ics)} KS references.\n")


if __name__ == "__main__":
    generate_burgers()
    generate_navier_stokes()
    generate_ks()
    print("All reference solutions generated.")
