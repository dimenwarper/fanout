"""2D Navier-Stokes Vorticity Equation Solver.

PDE:  omega_t + (u . grad)omega = nu * laplacian(omega)
      laplacian(psi) = -omega,  u = (psi_y, -psi_x)
Domain: [0, 2*pi] x [0, 2*pi], doubly periodic boundary conditions
Viscosity: nu = 1e-3
Grid: 64 x 64
Time: 10 trajectory snapshots up to t_final = 10.0

Scoring: nRMSE = ||pred - ref||_2 / ||ref||_2 over full trajectories,
averaged across 20 test instances. Score = 1/(1 + avg_nRMSE).
Runtime budget: 1.5 seconds for the full batch (20 instances).

Input:
  u0_batch: [batch_size, 64, 64] initial vorticity fields
  t_coordinates: [T] time points to return solution at (not including t=0)
  nu: kinematic viscosity (default 1e-3)

Output: numpy array of shape [batch_size, T, 64, 64].
"""

import numpy as np

NU = 1e-3
NX = 64
NY = 64
T_FINAL = 10.0
DOMAIN = (0.0, 2 * np.pi)


def solve_pde(u0_batch: np.ndarray, t_coordinates: np.ndarray, nu: float = NU) -> np.ndarray:
    """Solve 2D Navier-Stokes (vorticity-streamfunction) with explicit Euler.

    Parameters
    ----------
    u0_batch : np.ndarray
        Batch of initial vorticity fields, shape [batch_size, nx, nx].
    t_coordinates : np.ndarray
        Time points to record snapshots at, shape [T].
    nu : float
        Kinematic viscosity.

    Returns
    -------
    np.ndarray
        Vorticity trajectories of shape [batch_size, T, nx, nx].
    """
    batch_size = u0_batch.shape[0]
    nx = u0_batch.shape[1]
    n_times = len(t_coordinates)
    results = np.zeros((batch_size, n_times, nx, nx))
    dx = 2 * np.pi / nx

    for b in range(batch_size):
        omega = u0_batch[b].copy()
        t = 0.0
        dt = 0.05
        t_idx = 0

        while t_idx < n_times:
            target = t_coordinates[t_idx]
            while t < target - 1e-14:
                step = min(dt, target - t)

                # Solve Poisson via Jacobi iteration
                psi = np.zeros_like(omega)
                for _ in range(10):
                    psi = 0.25 * (
                        np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0)
                        + np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1)
                        + dx**2 * omega
                    )

                # Velocity from streamfunction
                u = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2 * dx)
                v = -(np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2 * dx)

                # Advection: central differences
                domega_dx = (np.roll(omega, -1, axis=0) - np.roll(omega, 1, axis=0)) / (2 * dx)
                domega_dy = (np.roll(omega, -1, axis=1) - np.roll(omega, 1, axis=1)) / (2 * dx)
                advection = u * domega_dx + v * domega_dy

                # Diffusion: 5-point Laplacian
                laplacian = (
                    np.roll(omega, -1, axis=0) + np.roll(omega, 1, axis=0)
                    + np.roll(omega, -1, axis=1) + np.roll(omega, 1, axis=1)
                    - 4 * omega
                ) / dx**2

                omega = omega + step * (-advection + nu * laplacian)
                t += step

            results[b, t_idx] = omega
            t_idx += 1

    return results
