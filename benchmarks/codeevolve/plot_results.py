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
        bench = r.get("benchmark")
        if bench is not None and not isinstance(bench, str):
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
    darw_scores = load_scores("diverse_model_set_darwinian")
    alpha_results = load_results("diverse_model_set_full_run")
    darw_results = load_results("diverse_model_set_darwinian")

    # AlphaEvolve final scores
    alpha_final: dict[str, float] = {}
    for r in alpha_results:
        alpha_final[r["task"]] = r["best_score"]

    # Darwinian final scores (from results.json — may differ from last csv step)
    darw_final: dict[str, float] = {}
    for r in darw_results:
        darw_final[r["task"]] = r["best_score"]

    # Benchmarks
    benchmarks: dict[str, float] = {}
    for r in alpha_results + darw_results:
        bench = r.get("benchmark")
        if bench is not None and not isinstance(bench, str):
            benchmarks[r["task"]] = bench

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CodeEvolve: AlphaEvolve vs Darwinian Progression (Diverse Model Set)", fontsize=14, fontweight="bold")

    for idx, task in enumerate(TASK_ORDER):
        ax = axes[idx // 2][idx % 2]

        # Darwinian step-by-step
        has_data = task in darw_scores
        if has_data:
            steps = np.array(darw_scores[task]["steps"])
            scores = np.array(darw_scores[task]["scores"])
            ax.plot(steps, scores, color=STRATEGY_COLORS["darwinian"], linewidth=2, label="Darwinian", zorder=3)
            ax.scatter(steps[-1], scores[-1], color=STRATEGY_COLORS["darwinian"], s=50, zorder=4)

        # AlphaEvolve as horizontal line (only final score known)
        if task in alpha_final:
            ax.axhline(
                alpha_final[task], color=STRATEGY_COLORS["alphaevolve"],
                linestyle="-", linewidth=2, alpha=0.8, label=f"AlphaEvolve ({alpha_final[task]:.4g})",
            )

        # Benchmark line
        if task in benchmarks:
            ax.axhline(
                benchmarks[task], color="black",
                linestyle="--", linewidth=1, alpha=0.5, label=f"Benchmark ({benchmarks[task]:.4g})",
            )

        ax.set_title(TASK_LABELS[task], fontsize=12)
        ax.set_xlabel("Step")
        ax.set_ylabel("Best Score")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)

        # Set y-axis with some padding
        ymin = min(0, scores.min() if has_data else 0)
        ymax_vals = [scores.max() if has_data else 0]
        if task in alpha_final:
            ymax_vals.append(alpha_final[task])
        if task in benchmarks:
            ymax_vals.append(benchmarks[task])
        ymax = max(ymax_vals)
        margin = (ymax - ymin) * 0.1
        ax.set_ylim(ymin - margin, ymax + margin)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "progression_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_DIR / 'progression_comparison.png'}")
    plt.close(fig)


if __name__ == "__main__":
    plot_strategy_comparison()
    plot_progression()
    print("Done.")
