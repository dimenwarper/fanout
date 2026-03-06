# PDE Solvers Benchmark

Evolutionary code generation for numerical PDE solvers. Uses fanout's multi-model search to evolve solvers that outperform simple baselines on classical PDEs.

## Tasks

| Task | PDE | Difficulty | Grid | Instances | Runtime Budget | Baseline Score |
|------|-----|------------|------|-----------|----------------|----------------|
| `burgers_1d` | 1D Burgers (nu=0.01, t=5) | Easy | 128 | 20 | 2s | ~0.65 |
| `navier_stokes_2d` | 2D Navier-Stokes vorticity (t=10) | Medium | 64x64 | 20 | 10s | ~0.56 |
| `ks_1d` | 1D Kuramoto-Sivashinsky (t=50) | Hard | 256 | 20 | 10s | ~0.27 |

## Metric

Following [CodePDE (arXiv:2505.08783)](https://arxiv.org/abs/2505.08783):

- **nRMSE** = `||pred - ref||_2 / ||ref||_2` (L2 norm ratio over full trajectory)
- **Score** = `1 / (1 + avg_nRMSE)` averaged across 20 instances per task
- Evaluation uses 10 trajectory snapshots (not just final state)
- **Runtime penalty**: solutions exceeding the time budget are penalized by `score *= budget / elapsed`

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

## Dependencies

Only `numpy` and `scipy` — both included in `[benchmarks]` optional deps.
