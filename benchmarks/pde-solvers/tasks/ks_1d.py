"""1D Kuramoto-Sivashinsky Equation Solver.

PDE:  u_t + u * u_x + u_xx + u_xxxx = 0
Domain: x in [0, 32*pi], periodic boundary conditions
Grid: 256 points
Time: 10 trajectory snapshots up to t_final = 50.0

The KS equation exhibits spatiotemporal chaos, making it a challenging
test for numerical PDE solvers. The fourth-order derivative u_xxxx
requires careful handling for stability.

Scoring: nRMSE = ||pred - ref||_2 / ||ref||_2 over full trajectories,
averaged across 20 test instances. Score = 1/(1 + avg_nRMSE).
Runtime budget: 10 seconds for the full batch (20 instances).

NOTE: The baseline explicit Euler solver exceeds the runtime budget and
receives a penalty. A better algorithm is needed to score well.

Input:
  u0_batch: [batch_size, 256] initial conditions
  t_coordinates: [T] time points to return solution at (not including t=0)

Output: numpy array of shape [batch_size, T, 256].
"""

import numpy as np

NX = 256
T_FINAL = 50.0
DOMAIN = (0.0, 32 * np.pi)


def solve_pde(u0_batch: np.ndarray, t_coordinates: np.ndarray) -> np.ndarray:
    """Solve 1D Kuramoto-Sivashinsky equation using explicit Euler.

    Parameters
    ----------
    u0_batch : np.ndarray
        Batch of initial conditions, shape [batch_size, nx].
    t_coordinates : np.ndarray
        Time points to record snapshots at, shape [T].

    Returns
    -------
    np.ndarray
        Solutions of shape [batch_size, T, nx].
    """
    batch_size, nx = u0_batch.shape
    n_times = len(t_coordinates)
    results = np.zeros((batch_size, n_times, nx))

    L = 32 * np.pi
    dx = L / nx
    dt = 0.01 * dx**4
    dt = min(dt, 0.01)

    for b in range(batch_size):
        u = u0_batch[b].copy()
        t = 0.0
        t_idx = 0

        while t_idx < n_times:
            target = t_coordinates[t_idx]
            while t < target - 1e-14:
                step = min(dt, target - t)

                u_right = np.roll(u, -1)
                u_left = np.roll(u, 1)
                u_right2 = np.roll(u, -2)
                u_left2 = np.roll(u, 2)

                ux = (u_right - u_left) / (2 * dx)
                uxx = (u_right - 2 * u + u_left) / dx**2
                uxxxx = (u_left2 - 4 * u_left + 6 * u - 4 * u_right + u_right2) / dx**4

                u = u + step * (-u * ux - uxx - uxxxx)
                t += step

            results[b, t_idx] = u
            t_idx += 1

    return results
