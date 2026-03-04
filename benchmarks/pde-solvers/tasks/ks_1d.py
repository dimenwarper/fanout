"""1D Kuramoto-Sivashinsky Equation Solver.

PDE: u_t + u * u_x + u_xx + u_xxxx = 0
Domain: x ∈ [0, 32π], periodic boundary conditions
Grid: 256 points
Time: integrate to t_final = 50.0

The KS equation exhibits spatiotemporal chaos, making it a challenging
test for numerical PDE solvers. The fourth-order derivative u_xxxx
requires careful handling for stability.

Objective: Evolve a numerical solver that produces accurate solutions
measured by nRMSE against a high-resolution ETDRK4 reference.

Benchmark (baseline explicit Euler): score ≈ 0.60  (nRMSE ~ 0.67)

Output: 1D numpy array of shape (256,) representing u(x, t_final).
"""

import numpy as np

NX = 256
T_FINAL = 50.0
DOMAIN = (0.0, 32 * np.pi)


def solve_pde(ic: np.ndarray, nx: int, t_final: float) -> np.ndarray:
    """Solve 1D Kuramoto-Sivashinsky equation using explicit Euler.

    Parameters
    ----------
    ic : np.ndarray
        Initial condition u(x, 0) on a uniform grid of size `nx`.
    nx : int
        Number of spatial grid points.
    t_final : float
        Final integration time.

    Returns
    -------
    np.ndarray
        Solution u(x, t_final) of shape (nx,).
    """
    L = 32 * np.pi
    dx = L / nx
    dt = 0.01 * dx**4  # very restrictive CFL for 4th-order term
    dt = min(dt, 0.01)
    u = ic.copy()
    t = 0.0

    while t < t_final:
        if t + dt > t_final:
            dt = t_final - t

        u_right = np.roll(u, -1)
        u_left = np.roll(u, 1)
        u_right2 = np.roll(u, -2)
        u_left2 = np.roll(u, 2)

        # u_x: central difference
        ux = (u_right - u_left) / (2 * dx)
        # u_xx: central second difference
        uxx = (u_right - 2 * u + u_left) / dx**2
        # u_xxxx: central fourth difference
        uxxxx = (u_left2 - 4 * u_left + 6 * u - 4 * u_right + u_right2) / dx**4

        u = u + dt * (-u * ux - uxx - uxxxx)
        t += dt

    return u
