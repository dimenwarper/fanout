#!/usr/bin/env python3
"""Plot MolOpt benchmark results.

Generates a progression plot showing step-by-step score evolution for each task,
comparing diverse vs small model set runs.

Usage:
    python benchmarks/molopt/plot_results.py
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
    "maximize_qed": "Maximize QED",
    "qed_logp_balance": "QED-LogP Balance",
    "constrained_generation": "Constrained Generation",
    "drug_candidate": "Drug Candidate",
}

TASK_ORDER = ["maximize_qed", "qed_logp_balance", "constrained_generation", "drug_candidate"]

RUN_COLORS = {
    "diverse_test": "#ED7D31",
    "small_test": "#5B9BD5",
}

RUN_LABELS = {
    "diverse_test": "Diverse Model Set",
    "small_test": "Small Model Set",
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
    run_names = ["diverse_test", "small_test"]
    all_scores = {r: load_scores(r) for r in run_names}

    # Benchmarks
    benchmarks: dict[str, float] = {}
    for r in run_names:
        for res in load_results(r):
            bench = res.get("benchmark")
            if bench is not None and not isinstance(bench, str):
                benchmarks[res["task"]] = bench

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MolOpt: Score Progression (Darwinian)", fontsize=14, fontweight="bold")

    for idx, task in enumerate(TASK_ORDER):
        ax = axes[idx // 2][idx % 2]
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
                if scores[-1] > 0:
                    ax.annotate(f"{scores[-1]:.4g}", (steps[-1], scores[-1]),
                                textcoords="offset points", xytext=(-40, 8),
                                fontsize=8, fontweight="bold", color=color)
                ymin_vals.append(scores.min())
                ymax_vals.append(scores.max())

        if task in benchmarks:
            ax.axhline(
                benchmarks[task], color="black",
                linestyle="--", linewidth=1, alpha=0.5,
                label=f"Benchmark ({benchmarks[task]:.4g})",
            )
            ymax_vals.append(benchmarks[task])

        ax.set_title(TASK_LABELS[task], fontsize=12)
        ax.set_xlabel("Solutions Submitted")
        ax.set_ylabel("Best Score")
        ax.legend(fontsize=9, loc="lower right")
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
