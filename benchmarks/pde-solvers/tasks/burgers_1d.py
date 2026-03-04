"""1D Burgers' Equation Solver.

PDE: u_t + u * u_x = ν * u_xx
Domain: x ∈ [0, 2π], periodic boundary conditions
Viscosity: ν = 0.01
Initial condition: u(x, 0) = sin(x)  (and variants for other test instances)
Grid: 128 points
Time: integrate to t_final = 5.0

Objective: Evolve a numerical solver that produces accurate solutions
measured by nRMSE against a high-resolution spectral reference.

Benchmark (baseline upwind Euler): score ≈ 0.70  (nRMSE ~ 0.43)

Output: 1D numpy array of shape (128,) representing u(x, t_final).
"""

import numpy as np

NU = 0.01
NX = 128
T_FINAL = 5.0
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

    while t < t_final:
        # Advective CFL only (first-order upwind is inherently diffusive)
        u_max = np.max(np.abs(u)) + 1e-10
        dt = 0.8 * dx / u_max
        if t + dt > t_final:
            dt = t_final - t

        u_right = np.roll(u, -1)
        u_left = np.roll(u, 1)

        # First-order upwind for advection + central diff for viscosity
        # Upwind: use backward diff where u>0, forward where u<0
        flux_pos = u * (u - u_left) / dx
        flux_neg = u * (u_right - u) / dx
        advection = np.where(u >= 0, flux_pos, flux_neg)
        diffusion = nu * (u_right - 2 * u + u_left) / dx**2
        u = u + dt * (-advection + diffusion)
        t += dt

    return u
