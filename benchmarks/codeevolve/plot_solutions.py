#!/usr/bin/env python3
"""Visualize winning CodeEvolve solutions.

Generates plots for each task's best solution from the alphaevolve full run.

Usage:
    python benchmarks/codeevolve/plot_solutions.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.spatial.distance import pdist, squareform

BENCH_DIR = Path(__file__).resolve().parent
RUN_DIR = BENCH_DIR / "runs" / "diverse_model_set_full_run"
SOL_DIR = RUN_DIR / "solutions"
OUT_DIR = BENCH_DIR / "plots"


def load_recorded_scores() -> dict[str, float]:
    """Load best scores from results.json (the actual benchmark scores)."""
    with open(RUN_DIR / "results.json") as f:
        results = json.load(f)
    return {r["task"]: r["best_score"] for r in results}

DARK_BG = "#0D1117"
PANEL_BG = "#161B22"
TEXT_COLOR = "#C9D1D9"
BORDER_COLOR = "#30363D"
ACCENT_BLUE = "#58A6FF"
ACCENT_ORANGE = "#F78166"
ACCENT_GREEN = "#3FB950"
ACCENT_PURPLE = "#BC8CFF"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("sol", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)


def plot_circle_packing(recorded_scores: dict[str, float]):
    mod = load_module(SOL_DIR / "circle_packing_1.py")
    result = mod.circle_packing26()  # (26, 3): x, y, r
    best_score = recorded_scores.get("circle_packing", np.sum(result[:, 2]))

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(DARK_BG)
    style_ax(ax)

    # Draw unit square
    square = mpatches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor=ACCENT_BLUE,
                                 facecolor="none", linestyle="-")
    ax.add_patch(square)

    # Draw circles
    total_radii = 0.0
    cmap = plt.cm.plasma
    radii = result[:, 2]
    r_min, r_max = radii.min(), radii.max()

    for i, (x, y, r) in enumerate(result):
        total_radii += r
        # Color by radius
        norm_r = (r - r_min) / (r_max - r_min + 1e-12)
        color = cmap(0.2 + 0.6 * norm_r)
        circle = mpatches.Circle((x, y), r, facecolor=(*color[:3], 0.4),
                                  edgecolor=(*color[:3], 0.9), linewidth=1.5)
        ax.add_patch(circle)
        # Radius text in center
        fontsize = max(5, min(9, r * 120))
        ax.text(x, y, f"{r:.3f}", ha="center", va="center",
                fontsize=fontsize, color=TEXT_COLOR, fontweight="bold")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(f"Circle Packing (26 circles)  |  Best sum of radii = {best_score:.4f}",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / "solution_circle_packing.png", dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_DIR / 'solution_circle_packing.png'}")
    plt.close(fig)


def plot_heilbronn_triangle(recorded_scores: dict[str, float]):
    mod = load_module(SOL_DIR / "heilbronn_triangle_1.py")
    points = mod.heilbronn_triangle11()  # (11, 2)

    # Compute all triangle areas
    from itertools import combinations
    n = len(points)
    tri_areas = []
    tri_indices = []
    for idx_triple in combinations(range(n), 3):
        i, j, k = idx_triple
        p1, p2, p3 = points[i], points[j], points[k]
        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        norm_area = area / (np.sqrt(3) / 4)
        tri_areas.append(norm_area)
        tri_indices.append(idx_triple)

    min_idx = np.argmin(tri_areas)
    min_area = tri_areas[min_idx]
    min_tri = tri_indices[min_idx]

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor(DARK_BG)
    style_ax(ax)

    # Draw bounding equilateral triangle
    tri_verts = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax.plot(tri_verts[:, 0], tri_verts[:, 1], color=ACCENT_BLUE, linewidth=2, alpha=0.7)

    # Highlight the minimum-area triangle
    min_pts = points[list(min_tri)]
    tri_patch = plt.Polygon(min_pts, facecolor=ACCENT_ORANGE, alpha=0.25,
                             edgecolor=ACCENT_ORANGE, linewidth=2)
    ax.add_patch(tri_patch)

    # Draw all points
    ax.scatter(points[:, 0], points[:, 1], s=80, color=ACCENT_GREEN, zorder=5,
               edgecolors="white", linewidths=0.5)

    # Label points
    for i, (x, y) in enumerate(points):
        ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(6, 6),
                    fontsize=9, color=TEXT_COLOR, fontweight="bold")

    # Highlight min triangle vertices
    ax.scatter(min_pts[:, 0], min_pts[:, 1], s=120, facecolors="none",
               edgecolors=ACCENT_ORANGE, linewidths=2, zorder=6)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, np.sqrt(3) / 2 + 0.05)
    ax.set_aspect("equal")
    best_score = recorded_scores.get("heilbronn_triangle", min_area)
    ax.set_title(f"Heilbronn Triangle (11 points)  |  Best min area = {best_score:.6f}",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=ACCENT_ORANGE, alpha=0.25, edgecolor=ACCENT_ORANGE,
                       label=f"Smallest triangle ({min_tri})"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right",
              facecolor=PANEL_BG, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "solution_heilbronn_triangle.png", dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_DIR / 'solution_heilbronn_triangle.png'}")
    plt.close(fig)


def plot_kissing_number(recorded_scores: dict[str, float]):
    mod = load_module(SOL_DIR / "kissing_number_1.py")
    points = mod.kissing_number11()  # (N, 11)
    best_n = int(recorded_scores.get("kissing_number", len(points)))

    n = len(points)
    norms = np.linalg.norm(points, axis=1)
    max_norm = norms.max()

    # Pairwise distance matrix
    dists = squareform(pdist(points.astype(float)))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    np.fill_diagonal(dists, 0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                              gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.3})
    fig.patch.set_facecolor(DARK_BG)

    # Left: distance matrix heatmap
    ax = axes[0]
    style_ax(ax)
    # Show a subsampled matrix if too large
    show_n = min(n, 100)
    show_dists = dists[:show_n, :show_n]
    im = ax.imshow(show_dists, cmap="viridis", aspect="auto")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors=TEXT_COLOR)
    cb.outline.set_edgecolor(BORDER_COLOR)
    cb.set_label("Euclidean Distance", color=TEXT_COLOR)
    label_suffix = f" (first {show_n})" if show_n < n else ""
    ax.set_title(f"Pairwise Distance Matrix{label_suffix}", fontsize=12, fontweight="bold", color=TEXT_COLOR)
    ax.set_xlabel("Point index")
    ax.set_ylabel("Point index")

    # Right: norm distribution + stats
    ax2 = axes[1]
    style_ax(ax2)
    ax2.hist(norms, bins=30, color=ACCENT_BLUE, alpha=0.7, edgecolor=DARK_BG)
    ax2.axvline(max_norm, color=ACCENT_ORANGE, linewidth=2, linestyle="--",
                label=f"Max norm = {max_norm:.2f}")
    ax2.axvline(min_dist, color=ACCENT_GREEN, linewidth=2, linestyle="--",
                label=f"Min pairwise dist = {min_dist:.2f}")
    ax2.set_title("Point Norm Distribution", fontsize=12, fontweight="bold", color=TEXT_COLOR)
    ax2.set_xlabel("Euclidean Norm")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=10, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR)

    # Stats text box
    ratio = min_dist / max_norm if max_norm > 0 else 0
    stats = (f"N = {n} points in 11D\n"
             f"Max norm = {max_norm:.4f}\n"
             f"Min pairwise dist = {min_dist:.4f}\n"
             f"Ratio (min_dist/max_norm) = {ratio:.4f}\n"
             f"Valid: {'Yes' if ratio >= 1.0 - 1e-9 else 'No'}")
    ax2.text(0.95, 0.95, stats, transform=ax2.transAxes, fontsize=9,
             verticalalignment="top", horizontalalignment="right",
             color=TEXT_COLOR, fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL_BG, edgecolor=BORDER_COLOR))

    fig.suptitle(f"Kissing Number in 11D  |  Best score = {best_n} points",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR, y=1.02)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "solution_kissing_number.png", dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_DIR / 'solution_kissing_number.png'}")
    plt.close(fig)


def plot_first_autocorr(recorded_scores: dict[str, float]):
    mod = load_module(SOL_DIR / "first_autocorr_1.py")
    a = mod.first_autocorrelation()  # 1D array

    n = len(a)
    b = np.convolve(a, a, "full")
    c1 = 2 * n * np.max(b) / (np.sum(a) ** 2)
    inv_c1 = 1.0 / c1

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(DARK_BG)

    # Top-left: the sequence a
    ax = axes[0, 0]
    style_ax(ax)
    ax.bar(range(n), a, color=ACCENT_BLUE, alpha=0.8, width=1.0, edgecolor=DARK_BG)
    ax.set_title("Sequence a", fontsize=12, fontweight="bold")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")

    # Top-right: autocorrelation b = conv(a, a)
    ax = axes[0, 1]
    style_ax(ax)
    ax.plot(range(len(b)), b, color=ACCENT_ORANGE, linewidth=1.5)
    ax.fill_between(range(len(b)), 0, b, color=ACCENT_ORANGE, alpha=0.15)
    peak_idx = np.argmax(b)
    ax.scatter([peak_idx], [b[peak_idx]], color=ACCENT_ORANGE, s=80, zorder=5)
    ax.annotate(f"max(b) = {b[peak_idx]:.4f}", (peak_idx, b[peak_idx]),
                textcoords="offset points", xytext=(10, 10),
                fontsize=9, color=ACCENT_ORANGE, fontweight="bold")
    ax.set_title("Autocorrelation b = conv(a, a)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Lag index")
    ax.set_ylabel("Value")

    # Bottom-left: sorted sequence values (distribution shape)
    ax = axes[1, 0]
    style_ax(ax)
    sorted_a = np.sort(a)[::-1]
    ax.plot(range(n), sorted_a, color=ACCENT_GREEN, linewidth=2)
    ax.fill_between(range(n), 0, sorted_a, color=ACCENT_GREEN, alpha=0.15)
    ax.set_title("Sorted Sequence Values (descending)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Value")

    # Bottom-right: stats panel
    ax = axes[1, 1]
    style_ax(ax)
    ax.axis("off")

    stats_lines = [
        f"Sequence length:  n = {n}",
        f"Sum of a:         {np.sum(a):.6f}",
        f"Max of b:         {np.max(b):.6f}",
        f"",
        f"C1 = 2n * max(b) / sum(a)^2",
        f"C1 = {c1:.6f}",
        f"",
        f"Score = 1/C1 = {inv_c1:.6f}",
    ]
    ax.text(0.1, 0.85, "\n".join(stats_lines), transform=ax.transAxes,
            fontsize=13, color=TEXT_COLOR, fontfamily="monospace",
            verticalalignment="top", linespacing=1.8,
            bbox=dict(boxstyle="round,pad=0.8", facecolor=PANEL_BG, edgecolor=BORDER_COLOR))

    best_score = recorded_scores.get("first_autocorr", inv_c1)
    fig.suptitle(f"First Autocorrelation  |  Best 1/C1 = {best_score:.6f}",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "solution_first_autocorr.png", dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_DIR / 'solution_first_autocorr.png'}")
    plt.close(fig)


if __name__ == "__main__":
    scores = load_recorded_scores()
    plot_circle_packing(scores)
    plot_heilbronn_triangle(scores)
    plot_kissing_number(scores)
    plot_first_autocorr(scores)
    print("Done.")
