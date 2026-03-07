"""1D Burgers' Equation Solver.

PDE:  u_t + u * u_x = nu * u_xx
Domain: x in [0, 2*pi], periodic boundary conditions
Viscosity: nu = 0.01
Grid: 128 points
Time: 10 trajectory snapshots up to t_final = 5.0

Scoring: nRMSE = ||pred - ref||_2 / ||ref||_2 over full trajectories,
averaged across 20 test instances. Score = 1/(1 + avg_nRMSE).
Runtime budget: 0.1 seconds for the full batch (20 instances).

Input:
  u0_batch: [batch_size, 128] initial conditions
  t_coordinates: [T] time points to return solution at (not including t=0)
  nu: viscosity coefficient (default 0.01)

Output: numpy array of shape [batch_size, T, 128].
"""

import numpy as np

NU = 0.01
NX = 128
T_FINAL = 5.0
DOMAIN = (0.0, 2 * np.pi)


def solve_pde(u0_batch: np.ndarray, t_coordinates: np.ndarray, nu: float = NU) -> np.ndarray:
    """Solve 1D Burgers equation using first-order upwind finite differences.

    Parameters
    ----------
    u0_batch : np.ndarray
        Batch of initial conditions, shape [batch_size, nx].
    t_coordinates : np.ndarray
        Time points to record snapshots at, shape [T].
    nu : float
        Viscosity coefficient.

    Returns
    -------
    np.ndarray
        Solutions of shape [batch_size, T, nx].
    """
    batch_size, nx = u0_batch.shape
    n_times = len(t_coordinates)
    results = np.zeros((batch_size, n_times, nx))
    dx = 2 * np.pi / nx

    for b in range(batch_size):
        u = u0_batch[b].copy()
        t = 0.0
        t_idx = 0

        while t_idx < n_times:
            target = t_coordinates[t_idx]
            while t < target - 1e-14:
                u_max = np.max(np.abs(u)) + 1e-10
                dt = 0.8 * dx / u_max
                dt = min(dt, target - t)

                u_right = np.roll(u, -1)
                u_left = np.roll(u, 1)

                flux_pos = u * (u - u_left) / dx
                flux_neg = u * (u_right - u) / dx
                advection = np.where(u >= 0, flux_pos, flux_neg)
                diffusion = nu * (u_right - 2 * u + u_left) / dx**2
                u = u + dt * (-advection + diffusion)
                t += dt

            results[b, t_idx] = u
            t_idx += 1

    return results
