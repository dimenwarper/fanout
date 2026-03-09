#!/usr/bin/env python3
"""Plot CodeEvolve benchmark results.

Generates two figures:
1. Strategy comparison bar chart (small model set: all strategies + darwinian)
2. Step-by-step progression (diverse model set: alphaevolve vs darwinian)

Usage:
    python benchmarks/codeevolve/plot_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import csv
import re

import matplotlib.pyplot as plt
import numpy as np

RUNS_DIR = Path(__file__).resolve().parent / "runs"
OUT_DIR = Path(__file__).resolve().parent / "plots"

TASK_LABELS = {
    "circle_packing": "Circle Packing",
    "kissing_number": "Kissing Number",
    "first_autocorr": "First Autocorrelation",
    "heilbronn_triangle": "Heilbronn Triangle",
}

TASK_ORDER = ["circle_packing", "kissing_number", "first_autocorr", "heilbronn_triangle"]

STRATEGY_COLORS = {
    "island": "#5B9BD5",
    "rsa": "#70AD47",
    "alphaevolve": "#ED7D31",
    "top-k": "#FFC000",
    "darwinian": "#C00000",
}


def parse_benchmark(value) -> float | None:
    """Extract a numeric benchmark from a value that may be a string like '1/C1 > 0.665'."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    # Try to extract the last number from a string (e.g. "1/C1 > 0.665" -> 0.665)
    matches = re.findall(r"[\d.]+", str(value))
    return float(matches[-1]) if matches else None


def load_results(run_name: str) -> list[dict]:
    path = RUNS_DIR / run_name / "results.json"
    with open(path) as f:
        return json.load(f)


def load_scores(run_name: str) -> dict[str, dict[str, list]]:
    """Load scores.csv and return {task: {"steps": [...], "scores": [...]}}."""
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


# ---------------------------------------------------------------------------
# Plot 1: Strategy comparison (small model set)
# ---------------------------------------------------------------------------


def plot_strategy_comparison():
    """Bar chart: best scores across strategies for small model set runs."""
    full = load_results("small_model_set_full_run")
    darw = load_results("small_model_set_darwinian")

    # Merge: full_run has island/rsa/alphaevolve/top-k, darwinian has darwinian
    all_results = full + darw

    # Collect best scores per (task, strategy)
    scores: dict[tuple[str, str], float] = {}
    benchmarks: dict[str, float] = {}
    for r in all_results:
        task = r["task"]
        strategy = r["strategy"]
        scores[(task, strategy)] = r["best_score"]
        bench = parse_benchmark(r.get("benchmark"))
        if bench is not None:
            benchmarks[task] = bench

    strategies = ["island", "rsa", "top-k", "alphaevolve", "darwinian"]
    n_tasks = len(TASK_ORDER)
    n_strats = len(strategies)

    fig, axes = plt.subplots(1, n_tasks, figsize=(16, 5), sharey=False)
    fig.suptitle("CodeEvolve: Strategy Comparison (Small Model Set)", fontsize=14, fontweight="bold", y=1.02)

    for i, task in enumerate(TASK_ORDER):
        ax = axes[i]
        vals = [scores.get((task, s), 0) for s in strategies]
        bars = ax.bar(
            range(n_strats), vals,
            color=[STRATEGY_COLORS[s] for s in strategies],
            edgecolor="white", linewidth=0.5,
        )

        # Benchmark line
        if task in benchmarks:
            ax.axhline(benchmarks[task], color="black", linestyle="--", linewidth=1, alpha=0.7, label="Benchmark")

        ax.set_title(TASK_LABELS[task], fontsize=11)
        ax.set_xticks(range(n_strats))
        ax.set_xticklabels(strategies, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Best Score" if i == 0 else "")

        # Add value labels on bars
        for bar, val in zip(bars, vals):
            if val > 0:
                label = f"{val:.0f}" if val > 10 else f"{val:.3f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    label, ha="center", va="bottom", fontsize=7,
                )

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [Patch(facecolor=STRATEGY_COLORS[s], label=s) for s in strategies]
    legend_elements.append(Line2D([0], [0], color="black", linestyle="--", label="Benchmark"))
    fig.legend(handles=legend_elements, loc="lower center", ncol=6, fontsize=9, bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / "strategy_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_DIR / 'strategy_comparison.png'}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Step-by-step progression (diverse model set: alphaevolve vs darwinian)
# ---------------------------------------------------------------------------


def plot_progression():
    """Line plots: step-by-step score progression, alphaevolve vs darwinian."""
    alpha_scores = load_scores("diverse_model_set_full_run")
    darw_scores = load_scores("diverse_model_set_darwinian")

    # Benchmarks from results.json
    benchmarks: dict[str, float] = {}
    for r in load_results("diverse_model_set_full_run") + load_results("diverse_model_set_darwinian"):
        bench = r.get("benchmark")
        if bench is not None and not isinstance(bench, str):
            benchmarks[r["task"]] = bench

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CodeEvolve: AlphaEvolve vs Darwinian Progression (Diverse Model Set)", fontsize=14, fontweight="bold")

    for idx, task in enumerate(TASK_ORDER):
        ax = axes[idx // 2][idx % 2]
        ymin_vals = [0.0]
        ymax_vals = [0.0]

        # AlphaEvolve step-by-step
        if task in alpha_scores:
            steps = np.array(alpha_scores[task]["steps"])
            scores = np.array(alpha_scores[task]["scores"])
            ax.plot(steps, scores, color=STRATEGY_COLORS["alphaevolve"], linewidth=2, label="AlphaEvolve", zorder=3)
            ax.scatter(steps[-1], scores[-1], color=STRATEGY_COLORS["alphaevolve"], s=50, zorder=4)
            ax.annotate(f"{scores[-1]:.4g}", (steps[-1], scores[-1]),
                        textcoords="offset points", xytext=(-40, 8),
                        fontsize=8, fontweight="bold", color=STRATEGY_COLORS["alphaevolve"])
            ymin_vals.append(scores.min())
            ymax_vals.append(scores.max())

        # Darwinian step-by-step
        if task in darw_scores:
            steps = np.array(darw_scores[task]["steps"])
            scores = np.array(darw_scores[task]["scores"])
            ax.plot(steps, scores, color=STRATEGY_COLORS["darwinian"], linewidth=2, label="Darwinian", zorder=3)
            ax.scatter(steps[-1], scores[-1], color=STRATEGY_COLORS["darwinian"], s=50, zorder=4)
            ax.annotate(f"{scores[-1]:.4g}", (steps[-1], scores[-1]),
                        textcoords="offset points", xytext=(-40, -14),
                        fontsize=8, fontweight="bold", color=STRATEGY_COLORS["darwinian"])
            ymin_vals.append(scores.min())
            ymax_vals.append(scores.max())

        # Benchmark line
        if task in benchmarks:
            ax.axhline(
                benchmarks[task], color="black",
                linestyle="--", linewidth=1, alpha=0.5, label=f"Benchmark ({benchmarks[task]:.4g})",
            )
            ymax_vals.append(benchmarks[task])

        ax.set_title(TASK_LABELS[task], fontsize=12)
        ax.set_xlabel("Step")
        ax.set_ylabel("Best Score")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)

        ymin = min(ymin_vals)
        ymax = max(ymax_vals)
        margin = (ymax - ymin) * 0.1 if ymax > ymin else 0.1
        ax.set_ylim(ymin - margin, ymax + margin)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "progression_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_DIR / 'progression_comparison.png'}")
    plt.close(fig)


if __name__ == "__main__":
    plot_strategy_comparison()
    plot_progression()
    print("Done.")
