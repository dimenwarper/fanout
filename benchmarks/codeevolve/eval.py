#!/usr/bin/env python3
"""CodeEvolve eval script for fanout.

Usage: ./eval.py <solution_file> [task_name]

task_name: circle_packing | kissing_number | first_autocorr | heilbronn_triangle

The solution file must define the task's entry function (e.g., circle_packing26()).
Prints a score (0.0-1.0) on the last line, based on benchmark_ratio.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np


def strip_code_fences(text: str) -> str:
    """Strip markdown code fences (```python ... ```) from LLM output."""
    # Remove opening fence with optional language tag
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    # Remove closing fence
    text = re.sub(r"\n?```$", "", text.strip())
    return text


def load_module(path: str):
    source = Path(path).read_text()
    cleaned = strip_code_fences(source)

    if cleaned != source:
        # Write cleaned version to a temp file so importlib can load it
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        tmp.write(cleaned)
        tmp.close()
        path = tmp.name

    spec = importlib.util.spec_from_file_location("solution", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Evaluators ──────────────────────────────────────────


def eval_circle_packing(sol) -> float:
    BENCHMARK = 2.6358627564136983
    TOL = 1e-6

    if not hasattr(sol, "circle_packing26"):
        print("Missing circle_packing26()", file=sys.stderr)
        return 0.0

    result = sol.circle_packing26()
    if not isinstance(result, np.ndarray) or result.shape != (26, 3):
        print(f"Bad shape: {getattr(result, 'shape', 'N/A')}", file=sys.stderr)
        return 0.0

    xs, ys, rs = result[:, 0], result[:, 1], result[:, 2]

    if np.any(rs < 0) or np.any(np.isnan(result)):
        print("Negative radii or NaN", file=sys.stderr)
        return 0.0

    if np.any(xs - rs < -TOL) or np.any(xs + rs > 1 + TOL):
        print("Circle outside square (x)", file=sys.stderr)
        return 0.0
    if np.any(ys - rs < -TOL) or np.any(ys + rs > 1 + TOL):
        print("Circle outside square (y)", file=sys.stderr)
        return 0.0

    for i, j in combinations(range(26), 2):
        dist = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
        if dist < rs[i] + rs[j] - TOL:
            print(f"Overlap between circles {i} and {j}", file=sys.stderr)
            return 0.0

    ratio = float(np.sum(rs)) / BENCHMARK
    print(f"sum_radii={np.sum(rs):.6f} benchmark_ratio={ratio:.4f}", file=sys.stderr)
    return min(1.0, ratio)


def eval_kissing_number(sol) -> float:
    BENCHMARK = 593

    if not hasattr(sol, "kissing_number11"):
        print("Missing kissing_number11()", file=sys.stderr)
        return 0.0

    points = sol.kissing_number11()
    if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 11:
        print(f"Bad shape: {getattr(points, 'shape', 'N/A')}", file=sys.stderr)
        return 0.0

    points = points.astype(int)
    n = len(points)

    norms = np.linalg.norm(points, axis=1)
    if np.any(norms == 0):
        print("Contains zero vector", file=sys.stderr)
        return 0.0

    max_norm = np.max(norms)

    min_dist = float("inf")
    for i, j in combinations(range(n), 2):
        d = np.linalg.norm(points[i] - points[j])
        min_dist = min(min_dist, d)

    if max_norm > min_dist + 1e-9:
        print(f"Constraint violated: max_norm={max_norm:.4f} > min_dist={min_dist:.4f}", file=sys.stderr)
        return 0.0

    ratio = n / BENCHMARK
    print(f"num_points={n} benchmark_ratio={ratio:.4f}", file=sys.stderr)
    return min(1.0, ratio)


def eval_first_autocorr(sol) -> float:
    BENCHMARK_INV_C1 = 0.6653

    if not hasattr(sol, "first_autocorrelation"):
        print("Missing first_autocorrelation()", file=sys.stderr)
        return 0.0

    a = sol.first_autocorrelation()
    if not isinstance(a, np.ndarray) or a.ndim != 1:
        print(f"Bad shape: {getattr(a, 'shape', 'N/A')}", file=sys.stderr)
        return 0.0

    if np.any(a < 0) or np.any(np.isnan(a)) or np.any(np.isinf(a)):
        print("Invalid values in sequence", file=sys.stderr)
        return 0.0

    s = np.sum(a)
    if s < 0.01:
        print("Sum too small", file=sys.stderr)
        return 0.0

    b = np.convolve(a, a, "full")
    c1 = 2 * len(a) * np.max(b) / (s**2)
    inv_c1 = 1.0 / c1

    ratio = inv_c1 / BENCHMARK_INV_C1
    print(f"C1={c1:.6f} inv_c1={inv_c1:.6f} benchmark_ratio={ratio:.4f}", file=sys.stderr)
    return min(1.0, ratio)


def eval_heilbronn(sol) -> float:
    BENCHMARK = 0.036529889880030156
    TOL = 1e-6

    if not hasattr(sol, "heilbronn_triangle11"):
        print("Missing heilbronn_triangle11()", file=sys.stderr)
        return 0.0

    points = sol.heilbronn_triangle11()
    if not isinstance(points, np.ndarray) or points.shape != (11, 2):
        print(f"Bad shape: {getattr(points, 'shape', 'N/A')}", file=sys.stderr)
        return 0.0

    h = np.sqrt(3) / 2
    for i, (x, y) in enumerate(points):
        if y < -TOL or y > h + TOL:
            print(f"Point {i} outside triangle (y)", file=sys.stderr)
            return 0.0
        if y > np.sqrt(3) * x + TOL or y > np.sqrt(3) * (1 - x) + TOL:
            print(f"Point {i} outside triangle", file=sys.stderr)
            return 0.0

    tri_area = h / 2
    min_area = float("inf")
    for i, j, k in combinations(range(11), 3):
        ax, ay = points[i]
        bx, by = points[j]
        cx, cy = points[k]
        area = abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) / 2.0
        normalized = area / tri_area
        min_area = min(min_area, normalized)

    ratio = min_area / BENCHMARK
    print(f"min_area_normalized={min_area:.8f} benchmark_ratio={ratio:.4f}", file=sys.stderr)
    return min(1.0, ratio)


EVALUATORS = {
    "circle_packing": eval_circle_packing,
    "kissing_number": eval_kissing_number,
    "first_autocorr": eval_first_autocorr,
    "heilbronn_triangle": eval_heilbronn,
}


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <solution_file> [task_name]", file=sys.stderr)
        print("0.0")
        sys.exit(0)

    solution_path = sys.argv[1]
    task_name = sys.argv[2] if len(sys.argv) > 2 else "circle_packing"

    try:
        sol = load_module(solution_path)
    except Exception as e:
        print(f"Load error: {e}", file=sys.stderr)
        print("0.0")
        sys.exit(0)

    if task_name not in EVALUATORS:
        print(f"Unknown task: {task_name}. Available: {list(EVALUATORS)}", file=sys.stderr)
        print("0.0")
        sys.exit(0)

    try:
        score = EVALUATORS[task_name](sol)
    except Exception as e:
        print(f"Eval error: {e}", file=sys.stderr)
        score = 0.0

    print(f"{score:.4f}")


if __name__ == "__main__":
    main()
