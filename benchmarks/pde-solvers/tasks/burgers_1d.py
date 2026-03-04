"""1D Burgers' Equation Solver.

PDE: u_t + u * u_x = ν * u_xx
Domain: x ∈ [0, 2π], periodic boundary conditions
Viscosity: ν = 0.01
Initial condition: u(x, 0) = sin(x)  (and variants for other test instances)
Grid: 128 points
Time: integrate to t_final = 1.0

Objective: Evolve a numerical solver that produces accurate solutions
measured by nRMSE against a high-resolution spectral reference.

Benchmark (baseline Lax-Friedrichs): score ≈ 0.84  (nRMSE ~ 0.19)

Output: 1D numpy array of shape (128,) representing u(x, t_final).
"""

import numpy as np

NU = 0.01
NX = 128
T_FINAL = 1.0
DOMAIN = (0.0, 2 * np.pi)


def solve_pde(ic: np.ndarray, nx: int, t_final: float, nu: float) -> np.ndarray:
    """Solve 1D Burgers equation using explicit Euler finite differences.

    Parameters
    ----------
    ic : np.ndarray
        Initial condition u(x, 0) on a uniform grid of size `nx`.
    nx : int
        Number of spatial grid points.
    t_final : float
        Final integration time.
    nu : float
        Viscosity coefficient.

    Returns
    -------
    np.ndarray
        Solution u(x, t_final) of shape (nx,).
    """
    dx = 2 * np.pi / nx
    u = ic.copy()
    t = 0.0

    # Coarse fixed timestep — stable but highly diffusive (Lax-Friedrichs)
    dt = 0.5 * dx  # large for a first-order method
    nsteps = max(int(np.ceil(t_final / dt)), 1)
    dt = t_final / nsteps

    for _ in range(nsteps):
        u_right = np.roll(u, -1)
        u_left = np.roll(u, 1)

        # Lax-Friedrichs scheme: u_new = 0.5*(u_left+u_right) - dt/(2dx)*(f_right-f_left)
        u = 0.5 * (u_left + u_right) - dt / (2 * dx) * (0.5 * u_right**2 - 0.5 * u_left**2) \
            + nu * dt / dx**2 * (u_right - 2 * u + u_left)

    return u
