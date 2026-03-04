#!/usr/bin/env python3
"""PDE Solvers eval script for fanout.

Usage: ./eval.py <solution_file> [task_name]

task_name: burgers_1d | navier_stokes_2d | ks_1d

The solution file must define solve_pde() with the appropriate signature.
Prints the score on the last line: score = 1.0 / (1.0 + avg_nrmse).
"""

from __future__ import annotations

import importlib.util
import signal
import sys
from pathlib import Path

import numpy as np

REF_DIR = Path(__file__).resolve().parent / "reference"


def load_module(path: str):
    spec = importlib.util.spec_from_file_location("solution", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def nrmse(pred: np.ndarray, ref: np.ndarray) -> float:
    """Normalized RMSE: RMSE / std(ref)."""
    std = np.std(ref)
    if std < 1e-12:
        return 0.0 if np.allclose(pred, ref) else 1e6
    return float(np.sqrt(np.mean((pred - ref) ** 2)) / std)


# ── Task configs ─────────────────────────────────────────────────────

TASK_CONFIGS = {
    "burgers_1d": {
        "instances": ["sin", "sin2", "gauss"],
        "nx": 128,
        "t_final": 5.0,
        "extra_args": {"nu": 0.01},
        "timeout": 120,
    },
    "navier_stokes_2d": {
        "instances": ["taylor_green", "double_shear", "random_modes"],
        "nx": 64,
        "t_final": 10.0,
        "extra_args": {"nu": 1e-3},
        "timeout": 300,
    },
    "ks_1d": {
        "instances": ["cos", "sin_modes", "localized"],
        "nx": 256,
        "t_final": 50.0,
        "extra_args": {},
        "timeout": 300,
    },
}


# ── Evaluator ────────────────────────────────────────────────────────


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Evaluation timed out")


def eval_task(sol, task_name: str) -> float:
    """Evaluate a solve_pde() function against reference solutions."""
    if not hasattr(sol, "solve_pde"):
        print("Missing solve_pde()", file=sys.stderr)
        return 0.0

    cfg = TASK_CONFIGS[task_name]
    instances = cfg["instances"]
    nx = cfg["nx"]
    t_final = cfg["t_final"]
    extra = cfg["extra_args"]
    timeout = cfg["timeout"]

    errors = []
    for inst_name in instances:
        ic_path = REF_DIR / f"{task_name}_ic_{inst_name}.npy"
        ref_path = REF_DIR / f"{task_name}_ref_{inst_name}.npy"

        if not ic_path.exists() or not ref_path.exists():
            print(f"Reference not found: {ic_path} or {ref_path}", file=sys.stderr)
            print("Run: python benchmarks/pde-solvers/reference/generate_references.py", file=sys.stderr)
            return 0.0

        ic = np.load(ic_path)
        ref = np.load(ref_path)

        try:
            # Set timeout
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)

            result = sol.solve_pde(ic, nx, t_final, **extra)

            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        except TimeoutError:
            print(f"  Instance '{inst_name}': TIMEOUT ({timeout}s)", file=sys.stderr)
            errors.append(1e6)
            continue
        except Exception as e:
            print(f"  Instance '{inst_name}': ERROR {e}", file=sys.stderr)
            errors.append(1e6)
            continue

        if not isinstance(result, np.ndarray):
            print(f"  Instance '{inst_name}': result not ndarray", file=sys.stderr)
            errors.append(1e6)
            continue

        if result.shape != ref.shape:
            print(f"  Instance '{inst_name}': shape mismatch {result.shape} vs {ref.shape}", file=sys.stderr)
            errors.append(1e6)
            continue

        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            print(f"  Instance '{inst_name}': NaN/Inf in result", file=sys.stderr)
            errors.append(1e6)
            continue

        err = nrmse(result, ref)
        errors.append(err)
        print(f"  Instance '{inst_name}': nRMSE={err:.6f}", file=sys.stderr)

    if not errors:
        return 0.0

    avg_nrmse = np.mean(errors)
    score = 1.0 / (1.0 + avg_nrmse)
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
