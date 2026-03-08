#!/usr/bin/env python3
"""Plot PDE Solvers benchmark results.

Generates a progression plot showing step-by-step score evolution for each task.

Usage:
    python benchmarks/pde-solvers/plot_results.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS_DIR = Path(__file__).resolve().parent / "runs"
OUT_DIR = Path(__file__).resolve().parent / "plots"

TASK_LABELS = {
    "burgers_1d": "Burgers 1D",
    "navier_stokes_2d": "Navier-Stokes 2D",
    "ks_1d": "Kuramoto-Sivashinsky 1D",
}

TASK_ORDER = ["burgers_1d", "navier_stokes_2d", "ks_1d"]

TASK_COLORS = {
    "burgers_1d": "#2176AE",
    "navier_stokes_2d": "#57B894",
    "ks_1d": "#F0803C",
}


def load_results(run_name: str) -> list[dict]:
    path = RUNS_DIR / run_name / "results.json"
    with open(path) as f:
        return json.load(f)


def load_scores(run_name: str) -> dict[str, dict[str, list]]:
    path = RUNS_DIR / run_name / "scores.csv"
    data: dict[str, dict[str, list]] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row["task"]
            if task not in data:
                data[task] = {"steps": [], "scores": []}
            data[task]["steps"].append(int(row["step"]))
            data[task]["scores"].append(float(row["max_score"]))
    return data


def plot_progression(run_name: str = "diverse_full"):
    scores = load_scores(run_name)
    results = load_results(run_name)

    benchmarks: dict[str, float] = {}
    for r in results:
        bench = r.get("benchmark")
        if bench is not None and not isinstance(bench, str):
            benchmarks[r["task"]] = bench

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"PDE Solvers: Score Progression ({run_name})", fontsize=14, fontweight="bold")

    for idx, task in enumerate(TASK_ORDER):
        ax = axes[idx]

        if task in scores:
            steps = np.array(scores[task]["steps"])
            vals = np.array(scores[task]["scores"])
            color = TASK_COLORS[task]

            ax.plot(steps, vals, color=color, linewidth=2.5, zorder=3)
            ax.fill_between(steps, 0, vals, color=color, alpha=0.1)
            ax.scatter(steps[-1], vals[-1], color=color, s=60, zorder=4)
            ax.annotate(
                f"{vals[-1]:.4f}", (steps[-1], vals[-1]),
                textcoords="offset points", xytext=(-40, 10),
                fontsize=9, fontweight="bold", color=color,
            )

        if task in benchmarks:
            ax.axhline(
                benchmarks[task], color="black",
                linestyle="--", linewidth=1, alpha=0.5,
                label=f"Baseline ({benchmarks[task]:.2f})",
            )

        ax.set_title(TASK_LABELS[task], fontsize=12)
        ax.set_xlabel("Step")
        ax.set_ylabel("Best Score")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.1)

    plt.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"progression_{run_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_progression("diverse_full")
    print("Done.")
