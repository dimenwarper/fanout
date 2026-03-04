# PDE Solvers Benchmark

Evolutionary code generation for numerical PDE solvers. Uses fanout's multi-model search to evolve solvers that outperform simple baselines on classical PDEs.

## Tasks

| Task | PDE | Difficulty | Grid | Baseline Score |
|------|-----|------------|------|----------------|
| `burgers_1d` | 1D Burgers (ν=0.01, t=5) | Easy | 128 | ~0.70 |
| `navier_stokes_2d` | 2D Navier-Stokes vorticity (t=10) | Medium | 64×64 | ~0.64 |
| `ks_1d` | 1D Kuramoto-Sivashinsky (t=50) | Hard | 256 | ~0.60 |

Score = `1 / (1 + avg_nRMSE)` where nRMSE is computed against high-resolution spectral reference solutions.

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

The baselines use explicit Euler finite differences — deliberately mediocre. LLMs can improve by:

- Using spectral/pseudo-spectral methods (FFT-based)
- Implicit or semi-implicit time integration (Crank-Nicolson, IMEX)
- Higher-order methods (RK4, ETDRK4)
- Operator splitting techniques
- Adaptive timestepping

## Dependencies

Only `numpy` and `scipy` — both included in `[benchmarks]` optional deps.
