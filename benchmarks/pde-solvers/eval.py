#!/usr/bin/env python3
"""PDE Solvers eval script for fanout.

Usage: ./eval.py <solution_file> [task_name]

task_name: burgers_1d | navier_stokes_2d | ks_1d

The solution file must define solve_pde(u0_batch, t_coordinates, [nu]) returning
a trajectory array of shape [batch_size, T, *spatial_dims].

Metric: CodePDE-style nRMSE = ||pred - ref||_2 / ||ref||_2 over full trajectories.
Score = 1.0 / (1.0 + avg_nrmse).

Runtime: Solutions exceeding the per-task time budget receive a score penalty.
"""

from __future__ import annotations

import importlib.util
import signal
import sys
import time
from pathlib import Path

import numpy as np

REF_DIR = Path(__file__).resolve().parent / "reference"

N_INSTANCES = 20


def load_module(path: str):
    spec = importlib.util.spec_from_file_location("solution", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def nrmse(pred: np.ndarray, ref: np.ndarray) -> float:
    """CodePDE-style nRMSE: ||pred - ref||_2 / ||ref||_2."""
    ref_norm = np.linalg.norm(ref.ravel())
    if ref_norm < 1e-12:
        return 0.0 if np.allclose(pred, ref) else 1e6
    return float(np.linalg.norm((pred - ref).ravel()) / ref_norm)


# -- Task configs --------------------------------------------------------------

TASK_CONFIGS = {
    "burgers_1d": {
        "n_instances": N_INSTANCES,
        "extra_args": {"nu": 0.01},
        "timeout": 120,
        "runtime_budget": 10.0,
    },
    "navier_stokes_2d": {
        "n_instances": N_INSTANCES,
        "extra_args": {"nu": 1e-3},
        "timeout": 300,
        "runtime_budget": 30.0,
    },
    "ks_1d": {
        "n_instances": N_INSTANCES,
        "extra_args": {},
        "timeout": 300,
        "runtime_budget": 30.0,
    },
}


# -- Evaluator -----------------------------------------------------------------


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Evaluation timed out")


def eval_task(sol, task_name: str) -> float:
    """Evaluate a solve_pde() function against reference solutions (batched)."""
    if not hasattr(sol, "solve_pde"):
        print("Missing solve_pde()", file=sys.stderr)
        return 0.0

    cfg = TASK_CONFIGS[task_name]
    n_inst = cfg["n_instances"]
    extra = cfg["extra_args"]
    timeout = cfg["timeout"]
    runtime_budget = cfg["runtime_budget"]

    # Load t_coordinates
    t_coords_path = REF_DIR / f"{task_name}_t_coordinates.npy"
    if not t_coords_path.exists():
        print(f"Missing {t_coords_path}", file=sys.stderr)
        print("Run: python benchmarks/pde-solvers/reference/generate_references.py", file=sys.stderr)
        return 0.0

    t_coordinates = np.load(t_coords_path)

    # Load all ICs and references
    ics = []
    refs = []
    for idx in range(n_inst):
        ic_path = REF_DIR / f"{task_name}_ic_{idx:02d}.npy"
        ref_path = REF_DIR / f"{task_name}_ref_{idx:02d}.npy"
        if not ic_path.exists() or not ref_path.exists():
            print(f"Reference not found: {ic_path} or {ref_path}", file=sys.stderr)
            print("Run: python benchmarks/pde-solvers/reference/generate_references.py", file=sys.stderr)
            return 0.0
        ics.append(np.load(ic_path))
        refs.append(np.load(ref_path))

    u0_batch = np.stack(ics, axis=0)  # [N, *spatial]
    ref_trajs = np.stack(refs, axis=0)  # [N, T, *spatial]

    # Run batched evaluation with timing
    try:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)

        t_start = time.monotonic()
        result = sol.solve_pde(u0_batch, t_coordinates, **extra)
        elapsed = time.monotonic() - t_start

        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    except TimeoutError:
        print(f"TIMEOUT ({timeout}s)", file=sys.stderr)
        return 0.0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 0.0

    print(f"  runtime: {elapsed:.2f}s (budget: {runtime_budget:.0f}s)", file=sys.stderr)

    if not isinstance(result, np.ndarray):
        print("Result is not ndarray", file=sys.stderr)
        return 0.0

    if result.shape != ref_trajs.shape:
        print(f"Shape mismatch: {result.shape} vs {ref_trajs.shape}", file=sys.stderr)
        return 0.0

    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
        print("NaN/Inf in result", file=sys.stderr)
        return 0.0

    # Compute per-instance nRMSE over full trajectory
    errors = []
    for i in range(n_inst):
        err = nrmse(result[i], ref_trajs[i])
        errors.append(err)
        print(f"  Instance {i:02d}: nRMSE={err:.6f}", file=sys.stderr)

    avg_nrmse = np.mean(errors)
    score = 1.0 / (1.0 + avg_nrmse)

    # Apply runtime penalty: if over budget, scale score by (budget / elapsed)
    if elapsed > runtime_budget:
        penalty = runtime_budget / elapsed
        penalized_score = score * penalty
        print(
            f"  OVER BUDGET: {elapsed:.1f}s > {runtime_budget:.0f}s, "
            f"penalty={penalty:.3f}, score {score:.4f} -> {penalized_score:.4f}",
            file=sys.stderr,
        )
        score = penalized_score

    print(f"avg_nRMSE={avg_nrmse:.6f} score={score:.6f}", file=sys.stderr)
    return score


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <solution_file> [task_name]", file=sys.stderr)
        print("0.0")
        sys.exit(0)

    solution_path = sys.argv[1]
    task_name = sys.argv[2] if len(sys.argv) > 2 else "burgers_1d"

    try:
        sol = load_module(solution_path)
    except Exception as e:
        print(f"Load error: {e}", file=sys.stderr)
        print("0.0")
        sys.exit(0)

    if task_name not in TASK_CONFIGS:
        print(f"Unknown task: {task_name}. Available: {list(TASK_CONFIGS)}", file=sys.stderr)
        print("0.0")
        sys.exit(0)

    try:
        score = eval_task(sol, task_name)
    except Exception as e:
        print(f"Eval error: {e}", file=sys.stderr)
        score = 0.0

    print(f"{score:.4f}")


if __name__ == "__main__":
    main()
