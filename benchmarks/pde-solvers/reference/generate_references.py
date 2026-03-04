#!/usr/bin/env python3
"""Generate high-accuracy reference solutions for PDE benchmark tasks.

Methods:
- Burgers 1D: Pseudo-spectral (FFT) at 512 points, downsampled to 128
- Navier-Stokes 2D: Pseudo-spectral (Fourier-Galerkin) at 256×256, downsampled to 64×64
- KS 1D: ETDRK4 at 1024 points, downsampled to 256

Each task generates 3 test instances with different initial conditions.
Reference solutions are stored as .npy files in this directory.

Usage:
    python benchmarks/pde-solvers/reference/generate_references.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REF_DIR = Path(__file__).resolve().parent


# ── Initial Conditions ───────────────────────────────────────────────


def burgers_ics(nx: int) -> list[tuple[str, np.ndarray]]:
    """Generate 3 initial conditions for 1D Burgers on [0, 2π]."""
    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    return [
        ("sin", np.sin(x)),
        ("sin2", np.sin(x) + 0.5 * np.sin(2 * x)),
        ("gauss", np.exp(-10 * (x - np.pi) ** 2)),
    ]


def ns_ics(nx: int) -> list[tuple[str, np.ndarray]]:
    """Generate 3 initial vorticity fields for 2D NS on [0, 2π]²."""
    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return [
        ("taylor_green", 2 * np.cos(X) * np.cos(Y)),
        ("double_shear", np.sin(2 * X) * np.cos(2 * Y) + 0.5 * np.cos(3 * X) * np.sin(3 * Y)),
        ("random_modes", (
            np.sin(X) * np.cos(2 * Y)
            + 0.8 * np.cos(3 * X) * np.sin(Y)
            + 0.5 * np.sin(2 * X) * np.sin(3 * Y)
        )),
    ]


def ks_ics(nx: int) -> list[tuple[str, np.ndarray]]:
    """Generate 3 initial conditions for 1D KS on [0, 32π]."""
    L = 32 * np.pi
    x = np.linspace(0, L, nx, endpoint=False)
    return [
        ("cos", np.cos(x / 16) * (1 + np.sin(x / 16))),
        ("sin_modes", np.sin(2 * np.pi * x / L) + 0.5 * np.sin(4 * np.pi * x / L)),
        ("localized", np.exp(-0.01 * (x - L / 2) ** 2) * np.cos(4 * np.pi * x / L)),
    ]


# ── Solvers ──────────────────────────────────────────────────────────


def solve_burgers_spectral(ic: np.ndarray, nx: int, t_final: float, nu: float) -> np.ndarray:
    """Pseudo-spectral solver for 1D Burgers with RK4 time integration."""
    dx = 2 * np.pi / nx
    k = np.fft.fftfreq(nx, d=dx / (2 * np.pi))
    k2 = k**2

    # Adaptive timestep
    dt = min(0.5 * dx / (np.max(np.abs(ic)) + 1e-10), 0.5 / (nu * np.max(k2) + 1e-10))
    dt = min(dt, 0.001)

    def rhs(u_hat):
        u = np.fft.ifft(u_hat).real
        nonlinear = -0.5 * 1j * k * np.fft.fft(u**2)
        diffusion = -nu * k2 * u_hat
        return nonlinear + diffusion

    u_hat = np.fft.fft(ic)
    t = 0.0
    while t < t_final:
        if t + dt > t_final:
            dt = t_final - t
        # RK4
        k1 = dt * rhs(u_hat)
        k2_ = dt * rhs(u_hat + 0.5 * k1)
        k3 = dt * rhs(u_hat + 0.5 * k2_)
        k4 = dt * rhs(u_hat + k3)
        u_hat = u_hat + (k1 + 2 * k2_ + 2 * k3 + k4) / 6
        t += dt

    return np.fft.ifft(u_hat).real


def solve_ns_spectral(ic: np.ndarray, nx: int, t_final: float, nu: float) -> np.ndarray:
    """Pseudo-spectral solver for 2D NS vorticity with RK4 + Fourier Poisson."""
    dx = 2 * np.pi / nx
    kx = np.fft.fftfreq(nx, d=dx / (2 * np.pi))
    ky = np.fft.fftfreq(nx, d=dx / (2 * np.pi))
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX**2 + KY**2
    K2_safe = K2.copy()
    K2_safe[0, 0] = 1.0

    # Dealias: 2/3 rule
    kmax = nx // 3
    dealias = np.ones((nx, nx))
    dealias[np.abs(KX) > kmax] = 0.0
    dealias[np.abs(KY) > kmax] = 0.0

    dt = min(0.5 * dx / (np.max(np.abs(ic)) + 1e-10), 0.25 * dx**2 / nu)
    dt = min(dt, 0.002)

    def rhs(omega_hat):
        # Streamfunction
        psi_hat = omega_hat / K2_safe
        psi_hat[0, 0] = 0.0
        # Velocity
        u_hat = 1j * KY * psi_hat
        v_hat = -1j * KX * psi_hat
        # Vorticity gradients
        domega_dx_hat = 1j * KX * omega_hat
        domega_dy_hat = 1j * KY * omega_hat
        # Transform to physical space
        u = np.fft.ifft2(u_hat).real
        v = np.fft.ifft2(v_hat).real
        domega_dx = np.fft.ifft2(domega_dx_hat).real
        domega_dy = np.fft.ifft2(domega_dy_hat).real
        # Advection in physical space, transform back and dealias
        advection = np.fft.fft2(u * domega_dx + v * domega_dy) * dealias
        # Diffusion in spectral space
        diffusion = -nu * K2 * omega_hat
        return -advection + diffusion

    omega_hat = np.fft.fft2(ic) * dealias
    t = 0.0
    while t < t_final:
        if t + dt > t_final:
            dt = t_final - t
        k1 = dt * rhs(omega_hat)
        k2_ = dt * rhs(omega_hat + 0.5 * k1)
        k3 = dt * rhs(omega_hat + 0.5 * k2_)
        k4 = dt * rhs(omega_hat + k3)
        omega_hat = (omega_hat + (k1 + 2 * k2_ + 2 * k3 + k4) / 6) * dealias
        t += dt

    return np.fft.ifft2(omega_hat).real


def solve_ks_etdrk4(ic: np.ndarray, nx: int, t_final: float) -> np.ndarray:
    """ETDRK4 solver for 1D Kuramoto-Sivashinsky equation.

    u_t = -u*u_x - u_xx - u_xxxx
    """
    L = 32 * np.pi
    dx = L / nx
    k = np.fft.fftfreq(nx, d=dx / (2 * np.pi)) * 2 * np.pi / (2 * np.pi / L)
    # Wavenumbers for domain [0, L]
    k = np.fft.fftfreq(nx, d=L / (2 * np.pi * nx))
    # Actually: k = 2π/L * fftfreq(nx)*nx
    k = 2 * np.pi / L * np.fft.fftfreq(nx) * nx

    # Linear operator: L = -k^2 + k^4  (but sign: u_t = -u*u_x - u_xx - u_xxxx)
    # => L = k^2 - k^4
    # Wait — KS equation: u_t + u*u_x + u_xx + u_xxxx = 0
    # => u_t = -u*u_x - u_xx - u_xxxx
    # In Fourier: d/dt û_k = -k^2 û_k - k^4 û_k - F(u*u_x)  -- NO
    # Actually in Fourier space: -u_xx -> k^2 (since u_xx -> -k^2 û, and -(-k^2)=k^2)
    # Wait let's be careful:
    # u_xx in Fourier: (ik)^2 û = -k^2 û
    # u_xxxx in Fourier: (ik)^4 û = k^4 û
    # So: û_t = -F(u*u_x) - (-k^2)û - k^4 û = -F(u*u_x) + k^2 û - k^4 û
    # Hmm that's wrong for KS. Let me re-derive.
    # KS: u_t = -u*u_x - u_xx - u_xxxx
    # Fourier of u_xx = -k² û, Fourier of u_xxxx = k⁴ û
    # So: û_t = -F(u·u_x) -(-k² û) - k⁴ û = -F(u·u_x) + k² û - k⁴ û
    # Linear part L_k = k² - k⁴
    Lk = k**2 - k**4

    dt = 0.5  # ETDRK4 allows larger timesteps
    nsteps = int(np.ceil(t_final / dt))
    dt = t_final / nsteps

    # ETDRK4 coefficients (Kassam & Trefethen, 2005)
    E = np.exp(Lk * dt)
    E2 = np.exp(Lk * dt / 2)

    # Contour integral for phi functions
    M = 32  # number of points on contour
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)

    # Reshape for broadcasting: Lk is (nx,), r is (M,)
    LR = dt * Lk[:, None] + r[None, :]  # (nx, M)

    Q = dt * np.real(np.mean(((np.exp(LR / 2) - 1) / LR), axis=1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1))
    f3 = dt * np.real(np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1))

    u_hat = np.fft.fft(ic)

    def nonlinear(u_hat):
        u = np.fft.ifft(u_hat).real
        return -0.5 * 1j * k * np.fft.fft(u**2)

    for _ in range(nsteps):
        Nu = nonlinear(u_hat)
        a = E2 * u_hat + Q * Nu
        Na = nonlinear(a)
        b = E2 * u_hat + Q * Na
        Nb = nonlinear(b)
        c = E2 * a + Q * (2 * Nb - Nu)
        Nc = nonlinear(c)
        u_hat = E * u_hat + Nu * f1 + 2 * (Na + Nb) * f2 + Nc * f3

    return np.fft.ifft(u_hat).real


# ── Downsampling ─────────────────────────────────────────────────────


def downsample_1d(u: np.ndarray, target_nx: int) -> np.ndarray:
    """Downsample 1D array via spectral truncation."""
    u_hat = np.fft.fft(u)
    n = len(u)
    nt = target_nx
    u_hat_trunc = np.zeros(nt, dtype=complex)
    # Copy low frequencies
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
    # Copy the four quadrants of low frequencies
    u_hat_trunc[:half, :half] = u_hat[:half, :half]
    u_hat_trunc[:half, -half:] = u_hat[:half, -half:]
    u_hat_trunc[-half:, :half] = u_hat[-half:, :half]
    u_hat_trunc[-half:, -half:] = u_hat[-half:, -half:]
    return np.fft.ifft2(u_hat_trunc).real * (nt / n) ** 2


# ── Main ─────────────────────────────────────────────────────────────


def generate_burgers():
    """Generate Burgers reference solutions at 512 points, downsample to 128."""
    print("Generating Burgers 1D references...")
    nx_hi = 512
    nx_lo = 128
    nu = 0.01
    t_final = 1.0

    ics = burgers_ics(nx_hi)
    refs = {}
    for name, ic in ics:
        print(f"  Instance '{name}'...", end=" ", flush=True)
        sol_hi = solve_burgers_spectral(ic, nx_hi, t_final, nu)
        sol_lo = downsample_1d(sol_hi, nx_lo)
        refs[name] = sol_lo

        # Also save the IC at low resolution for eval
        ic_lo = downsample_1d(ic, nx_lo)
        np.save(REF_DIR / f"burgers_1d_ic_{name}.npy", ic_lo)
        np.save(REF_DIR / f"burgers_1d_ref_{name}.npy", sol_lo)
        print(f"done (max={np.max(np.abs(sol_lo)):.4f})")

    print(f"  Saved {len(refs)} Burgers references.\n")


def generate_navier_stokes():
    """Generate NS reference solutions at 256×256, downsample to 64×64."""
    print("Generating Navier-Stokes 2D references...")
    nx_hi = 256
    nx_lo = 64
    nu = 1e-3
    t_final = 1.0

    ics = ns_ics(nx_hi)
    refs = {}
    for name, ic in ics:
        print(f"  Instance '{name}'...", end=" ", flush=True)
        sol_hi = solve_ns_spectral(ic, nx_hi, t_final, nu)
        sol_lo = downsample_2d(sol_hi, nx_lo)
        refs[name] = sol_lo

        ic_lo = downsample_2d(ic, nx_lo)
        np.save(REF_DIR / f"navier_stokes_2d_ic_{name}.npy", ic_lo)
        np.save(REF_DIR / f"navier_stokes_2d_ref_{name}.npy", sol_lo)
        print(f"done (max={np.max(np.abs(sol_lo)):.4f})")

    print(f"  Saved {len(refs)} Navier-Stokes references.\n")


def generate_ks():
    """Generate KS reference solutions at 1024 points, downsample to 256."""
    print("Generating Kuramoto-Sivashinsky 1D references...")
    nx_hi = 1024
    nx_lo = 256
    t_final = 50.0

    ics = ks_ics(nx_hi)
    refs = {}
    for name, ic in ics:
        print(f"  Instance '{name}'...", end=" ", flush=True)
        sol_hi = solve_ks_etdrk4(ic, nx_hi, t_final)
        sol_lo = downsample_1d(sol_hi, nx_lo)
        refs[name] = sol_lo

        ic_lo = downsample_1d(ic, nx_lo)
        np.save(REF_DIR / f"ks_1d_ic_{name}.npy", ic_lo)
        np.save(REF_DIR / f"ks_1d_ref_{name}.npy", sol_lo)
        print(f"done (max={np.max(np.abs(sol_lo)):.4f})")

    print(f"  Saved {len(refs)} KS references.\n")


if __name__ == "__main__":
    generate_burgers()
    generate_navier_stokes()
    generate_ks()
    print("All reference solutions generated.")
