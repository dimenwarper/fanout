#!/usr/bin/env python3
"""Plot MNIST Weights benchmark results.

Generates a progression plot showing step-by-step score evolution,
comparing test vs test_2 runs.

Usage:
    python benchmarks/mnist-weights/plot_results.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS_DIR = Path(__file__).resolve().parent / "runs"
OUT_DIR = Path(__file__).resolve().parent / "plots"

RUN_COLORS = {
    "test": "#5B9BD5",
    "test_2": "#ED7D31",
}

RUN_LABELS = {
    "test": "Run 1",
    "test_2": "Run 2",
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


def plot_progression():
    run_names = ["test", "test_2"]
    all_scores = {r: load_scores(r) for r in run_names}

    # Benchmarks
    benchmarks: dict[str, float] = {}
    for r in run_names:
        for res in load_results(r):
            bench = res.get("benchmark")
            if bench is not None and not isinstance(bench, str):
                benchmarks[res["task"]] = bench

    # Single task: classify_all
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("MNIST Weights: Score Progression (Darwinian)", fontsize=14, fontweight="bold")

    task = "classify_all"
    ymin_vals = [0.0]
    ymax_vals = [0.0]

    for run_name in run_names:
        scores_data = all_scores[run_name]
        if task in scores_data:
            steps = np.array(scores_data[task]["steps"])
            scores = np.array(scores_data[task]["scores"])
            color = RUN_COLORS[run_name]
            label = RUN_LABELS[run_name]
            ax.plot(steps, scores, color=color, linewidth=2, label=label, zorder=3)
            ax.scatter(steps[-1], scores[-1], color=color, s=50, zorder=4)
            ax.annotate(f"{scores[-1]:.4g}", (steps[-1], scores[-1]),
                        textcoords="offset points", xytext=(-40, 8),
                        fontsize=9, fontweight="bold", color=color)
            ymin_vals.append(scores.min())
            ymax_vals.append(scores.max())

    if task in benchmarks:
        ax.axhline(
            benchmarks[task], color="black",
            linestyle="--", linewidth=1, alpha=0.5,
            label=f"Benchmark ({benchmarks[task]:.4g})",
        )
        ymax_vals.append(benchmarks[task])

    ax.set_title("Classify All Digits", fontsize=12)
    ax.set_xlabel("Step")
    ax.set_ylabel("Best Score (Accuracy)")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)

    ymin = min(ymin_vals)
    ymax = max(ymax_vals)
    margin = (ymax - ymin) * 0.1 if ymax > ymin else 0.1
    ax.set_ylim(ymin - margin, ymax + margin)

    plt.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "progression.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_progression()
    print("Done.")
