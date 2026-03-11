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
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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


def get_test_data():
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    return X_test, y_test


def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(np.clip(x, -50, 0)) - 1))


def forward(weights, X):
    """Flexible forward pass — infers shapes from the weight dict."""
    W1 = np.array(weights["W1"], dtype=np.float32)
    b1 = np.array(weights["b1"], dtype=np.float32)
    W2 = np.array(weights["W2"], dtype=np.float32)
    b2 = np.array(weights["b2"], dtype=np.float32)
    h = elu(X @ W1 + b1)
    return h @ W2 + b2


def plot_solution(sol_name: str, weights: dict, subtitle: str):
    X_test, y_test = get_test_data()
    W1 = np.array(weights["W1"], dtype=np.float32)
    n_hidden = W1.shape[1]

    # Run predictions
    logits = forward(weights, X_test)
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y_test)
    cm = confusion_matrix(y_test, preds, labels=range(10))

    # Layout: top row = W1 feature maps, bottom row = confusion matrix
    n_cols = min(n_hidden, 16)
    fig_w = max(12, n_cols * 1.2)

    fig = plt.figure(figsize=(fig_w, 7))
    fig.patch.set_facecolor(DARK_BG)

    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1.3], hspace=0.35)

    # ── Top: W1 columns as 8×8 feature maps ──
    gs_top = gridspec.GridSpecFromSubplotSpec(1, n_cols, subplot_spec=gs[0], wspace=0.08)
    vmax = np.abs(W1).max()
    for i in range(n_cols):
        ax = fig.add_subplot(gs_top[0, i])
        ax.imshow(W1[:, i].reshape(8, 8), cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"h{i}", fontsize=7, color=TEXT_COLOR, pad=2)
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOR)
            spine.set_linewidth(0.5)

    # ── Bottom: confusion matrix ──
    ax_cm = fig.add_subplot(gs[1])
    style_ax(ax_cm)
    im = ax_cm.imshow(cm, cmap="Blues", interpolation="nearest")

    # Annotate cells
    for i in range(10):
        for j in range(10):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else TEXT_COLOR
            ax_cm.text(j, i, str(val), ha="center", va="center",
                       fontsize=8, color=color, fontweight="bold" if i == j else "normal")

    ax_cm.set_xticks(range(10))
    ax_cm.set_yticks(range(10))
    ax_cm.set_xlabel("Predicted", fontsize=10)
    ax_cm.set_ylabel("True", fontsize=10)
    ax_cm.set_title(f"Confusion Matrix  |  Accuracy = {acc:.1%}", fontsize=11,
                    fontweight="bold", pad=8)

    cb = fig.colorbar(im, ax=ax_cm, fraction=0.03, pad=0.02)
    cb.ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    cb.outline.set_edgecolor(BORDER_COLOR)

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
