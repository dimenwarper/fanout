#!/usr/bin/env python3
"""Visualize winning MNIST-Weights solutions from the test run.

For each solution: shows W1 columns as 8x8 learned features, the hand-crafted
templates or centroids, and a confusion matrix from running predictions on
actual sklearn digits data.

Usage:
    uv run python benchmarks/mnist-weights/plot_solutions.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
SOL_DIR = BENCH_DIR / "runs" / "test" / "solutions"
OUT_DIR = BENCH_DIR / "plots"

# ── Theme ──────────────────────────────────────────
DARK_BG = "#0D1117"
PANEL_BG = "#161B22"
TEXT_COLOR = "#C9D1D9"
BORDER_COLOR = "#30363D"
ACCENT_BLUE = "#58A6FF"
ACCENT_ORANGE = "#F78166"
ACCENT_GREEN = "#3FB950"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("sol", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)


def plot_solution(sol_name: str, weights: dict, subtitle: str):
    W1 = np.array(weights["W1"], dtype=np.float32)
    n_hidden = W1.shape[1]

    # Layout: 4-column grid of feature maps
    n_feat = min(n_hidden, 16)
    grid_rows = int(np.ceil(n_feat / 4))

    fig = plt.figure(figsize=(8, 2.2 * grid_rows + 0.8))
    fig.patch.set_facecolor(DARK_BG)

    gs = gridspec.GridSpec(grid_rows, 4, figure=fig, wspace=0.12, hspace=0.3)
    vmax = np.abs(W1).max()
    for i in range(n_feat):
        r, c = divmod(i, 4)
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(W1[:, i].reshape(8, 8), cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"h{i}", fontsize=8, color=TEXT_COLOR, pad=2)
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOR)
            spine.set_linewidth(0.5)

    fig.suptitle(f"MNIST Weights: {sol_name}\n{subtitle}",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR, y=1.02)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"solution_{sol_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"  Saved {out_path}")
    plt.close(fig)


SOLUTIONS = [
    ("classify_all_1", "classify_all_1.py", "Hardcoded pre-trained weights"),
    ("classify_all_2", "classify_all_2.py", "Centroid-based prototype matching"),
    ("classify_all_3", "classify_all_3.py", "Hand-crafted digit templates"),
]


def main():
    for sol_name, filename, subtitle in SOLUTIONS:
        path = SOL_DIR / filename
        if not path.exists():
            print(f"Skipping {sol_name}: {path} not found")
            continue
        print(f"Plotting {sol_name}...")
        mod = load_module(path)
        weights = mod.classify_all()
        plot_solution(sol_name, weights, subtitle)
    print("Done.")


if __name__ == "__main__":
    main()
