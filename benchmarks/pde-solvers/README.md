# PDE Solvers Benchmark

Evolutionary code generation for numerical PDE solvers. Uses fanout's multi-model search to evolve solvers that outperform simple baselines on classical PDEs.

## Tasks

| Task | PDE | Difficulty | Grid | Instances | Runtime Budget | Baseline Score |
|------|-----|------------|------|-----------|----------------|----------------|
| `burgers_1d` | 1D Burgers (nu=0.01, t=5) | Easy | 128 | 20 | 0.1s | ~0.65 |
| `navier_stokes_2d` | 2D Navier-Stokes vorticity (t=10) | Medium | 64x64 | 20 | 1.5s | ~0.56 |
| `ks_1d` | 1D Kuramoto-Sivashinsky (64π, t=200) | Hard | 256 | 20 | 10s | ~0.15 |

## Metric

Following [CodePDE (arXiv:2505.08783)](https://arxiv.org/abs/2505.08783):

- **nRMSE** = `||pred - ref||_2 / ||ref||_2` (L2 norm ratio over full trajectory)
- **Score** = `1 / (1 + avg_nRMSE)` averaged across 20 instances per task
- Evaluation uses 10 trajectory snapshots (not just final state)
- **Runtime penalty**: solutions exceeding the time budget are penalized by `score *= (budget / elapsed)²`

## Solver Interface

```python
def solve_pde(u0_batch, t_coordinates, nu=...):
    """
    Args:
        u0_batch: [batch_size, *spatial_dims]  (Burgers/KS: [B,N], NS: [B,N,N])
        t_coordinates: [T] time points to return solution at (NOT including t=0)
        nu: viscosity (not used by KS)
    Returns:
        solutions: [batch_size, T, *spatial_dims]
    """
```

## Quick Start

```bash
# 1. Generate reference solutions (one-time)
python benchmarks/pde-solvers/reference/generate_references.py

# 2. Verify eval works against baseline
python benchmarks/pde-solvers/eval.py benchmarks/pde-solvers/tasks/burgers_1d.py burgers_1d

# 3. Run the benchmark
uv run --extra benchmarks python benchmarks/pde-solvers/run_benchmark.py --tasks burgers_1d -r 1 -n 2 -m openai/gpt-4o-mini
```

## What LLMs Can Improve

The baselines use explicit Euler finite differences — deliberately mediocre. LLMs can improve by using better numerical methods, but must stay within the runtime budget (no brute-force upsampling).

## Benchmark Hardening Log

We iteratively tightened the benchmark as frontier models found ways to score near-perfectly:

| Version | Change | Burgers | NS | KS | Problem |
|---------|--------|---------|----|----|---------|
| v0 | Initial: 3 instances, final-state eval, range-based nRMSE | — | — | — | Too few instances, easy metric |
| v1 | CodePDE alignment: 20 instances, 10 trajectory snapshots, L2 nRMSE, batched eval | 0.65 | 0.56 | 0.27 | Baseline scores calibrated |
| v2 | Strip method hints from task files (no "spectral", "FFT", etc.) | — | — | — | LLMs were copying reference algorithms |
| v3 | Add runtime budgets (10s, 30s, 30s) + linear penalty | 0.65 | 0.56 | 0.27 | Prevent brute-force upsampling |
| v4 | Tighten budgets (2s, 10s, 10s) | 0.65 | 0.56 | 0.27 | Upsampled solutions (128→512, 64→128) still fit in budget |
| v5 | Tighten further (0.25s, 3s, 10s) + quadratic penalty `(budget/elapsed)²` | 0.65 | 0.56 | 0.15 | Upsampling now penalized below baseline; KS still trivial at native res |
| v6 | KS: domain 32π→64π, t_final 50→200 | 0.65 | 0.56 | 0.15 | Even perfect ETDRK4 at 256pts scores ~0.51 due to resolution gap vs 1024pt reference |
| v7 | Tighten Burgers (0.1s) and NS (1.5s) | 0.65 | 0.56 | 0.15 | v6 diverse run: models used 2x upsample (0.14s) and 3/2 padding (2.09s) within budget, scoring 0.997/0.987 |

**Key insight**: The hardest lever for spectral PDE solvers isn't the runtime budget — it's the gap between the evaluation grid and the reference grid. On a larger domain with the same grid, there aren't enough points to resolve all active chaotic modes, so even the "right" algorithm accumulates error.

## Dependencies

Only `numpy` and `scipy` — both included in `[benchmarks]` optional deps.
