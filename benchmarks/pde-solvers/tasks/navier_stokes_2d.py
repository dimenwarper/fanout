"""2D Navier-Stokes Vorticity Equation Solver.

PDE: ω_t + (u · ∇)ω = ν * ∇²ω
     ∇²ψ = -ω,  u = (ψ_y, -ψ_x)
Domain: [0, 2π] × [0, 2π], doubly periodic boundary conditions
Viscosity: ν = 1e-3
Grid: 64 × 64
Time: integrate to t_final = 10.0

The vorticity-streamfunction formulation is used. The streamfunction ψ is
recovered from vorticity ω via the Poisson equation, then velocity (u, v)
is computed from ψ.

Objective: Evolve a numerical solver that produces accurate vorticity fields
measured by nRMSE against a high-resolution pseudo-spectral reference.

Benchmark (baseline Jacobi-Euler): score ≈ 0.64  (nRMSE ~ 0.57)

Output: 2D numpy array of shape (64, 64) representing ω(x, y, t_final).
"""

import numpy as np

NU = 1e-3
NX = 64
NY = 64
T_FINAL = 10.0
DOMAIN = (0.0, 2 * np.pi)


def solve_pde(ic: np.ndarray, nx: int, t_final: float, nu: float) -> np.ndarray:
    """Solve 2D Navier-Stokes (vorticity-streamfunction) with explicit Euler.

    Parameters
    ----------
    ic : np.ndarray
        Initial vorticity field ω(x, y, 0) of shape (nx, nx).
    nx : int
        Number of grid points in each direction.
    t_final : float
        Final integration time.
    nu : float
        Kinematic viscosity.

    Returns
    -------
    np.ndarray
        Vorticity field ω(x, y, t_final) of shape (nx, nx).
    """
    dx = 2 * np.pi / nx
    dt = 0.05  # large fixed timestep
    omega = ic.copy()
    t = 0.0

    while t < t_final:
        if t + dt > t_final:
            dt = t_final - t

        # Solve Poisson ∇²ψ = -ω via Jacobi iteration (deliberately crude)
        psi = np.zeros_like(omega)
        for _ in range(10):  # very few Jacobi iterations — inaccurate
            psi = 0.25 * (
                np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0)
                + np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1)
                + dx**2 * omega
            )

        # Velocity from streamfunction: u = ψ_y, v = -ψ_x
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

        omega = omega + dt * (-advection + nu * laplacian)
        t += dt

    return omega
