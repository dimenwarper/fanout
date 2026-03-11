#!/usr/bin/env python3
"""Visualize winning MolOpt solutions from the diverse_test run.

For each task: a large molecule grid of all 100 molecules, with a smaller
row of property histograms below.

Usage:
    uv run python benchmarks/molopt/plot_solutions.py
"""

from __future__ import annotations

import importlib.util
import json
import statistics
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, QED as QEDModule

BENCH_DIR = Path(__file__).resolve().parent
RUN_DIR = BENCH_DIR / "runs" / "diverse_test"
SOL_DIR = RUN_DIR / "solutions"
OUT_DIR = BENCH_DIR / "plots"

# ── Theme ──────────────────────────────────────────
DARK_BG = "#0D1117"
PANEL_BG = "#161B22"
TEXT_COLOR = "#C9D1D9"
BORDER_COLOR = "#30363D"
ACCENT_BLUE = "#58A6FF"
ACCENT_ORANGE = "#F78166"
ACCENT_GREEN = "#3FB950"
ACCENT_PURPLE = "#BC8CFF"

TASKS = {
    "maximize_qed": {
        "func": "maximize_qed",
        "solution": "maximize_qed_1.py",
        "props": ["QED", "LogP", "MW"],
        "benchmark": 0.9,
    },
    "qed_logp_balance": {
        "func": "qed_logp_balance",
        "solution": "qed_logp_balance_1.py",
        "props": ["QED", "LogP", "MW"],
        "benchmark": 0.85,
    },
    "constrained_generation": {
        "func": "constrained_generation",
        "solution": "constrained_generation_1.py",
        "props": ["QED", "LogP", "MW", "TPSA", "Rings"],
        "benchmark": 0.85,
    },
    "drug_candidate": {
        "func": "drug_candidate",
        "solution": "drug_candidate_1.py",
        "props": ["QED", "LogP", "MW", "HBD", "HBA"],
        "benchmark": 0.85,
    },
}

PROP_THRESHOLDS = {
    "maximize_qed": {"QED": [(0.9, "benchmark")]},
    "qed_logp_balance": {"QED": [(0.8, "target")], "LogP": [(1.0, "low"), (3.0, "high")]},
    "constrained_generation": {
        "QED": [(0.75, "min")],
        "MW": [(250, "min"), (400, "max")],
        "LogP": [(1.5, "min"), (3.5, "max")],
        "TPSA": [(40, "min"), (90, "max")],
        "Rings": [(2, "min"), (4, "max")],
    },
    "drug_candidate": {
        "QED": [(0.7, "min")],
        "MW": [(200, "min"), (500, "max")],
        "LogP": [(0, "min"), (5, "max")],
        "HBD": [(5, "max")],
        "HBA": [(10, "max")],
    },
}


# ── Helpers ──────────────────────────────────────────

def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("sol", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_recorded_scores() -> dict[str, float]:
    p = RUN_DIR / "results.json"
    if p.exists():
        with open(p) as f:
            return {r["task"]: r["best_score"] for r in json.load(f)}
    return {}


def style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)


def compute_properties(mols: list) -> dict[str, list[float]]:
    props = {"QED": [], "LogP": [], "MW": [], "HBD": [], "HBA": [], "TPSA": [], "Rings": []}
    for mol in mols:
        props["QED"].append(QEDModule.qed(mol))
        props["LogP"].append(Descriptors.MolLogP(mol))
        props["MW"].append(Descriptors.MolWt(mol))
        props["HBD"].append(Descriptors.NumHDonors(mol))
        props["HBA"].append(Descriptors.NumHAcceptors(mol))
        props["TPSA"].append(Descriptors.TPSA(mol))
        props["Rings"].append(mol.GetRingInfo().NumRings())
    return props


def mol_grid_to_array(mols, molsPerRow=10, subImgSize=(200, 150)):
    """Render molecule grid to numpy array."""
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=molsPerRow,
        subImgSize=subImgSize,
        useSVG=False,
    )
    if isinstance(img, Image.Image):
        return np.array(img)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return np.array(Image.open(buf))


# ── Plotting ──────────────────────────────────────────

def plot_task(task_name: str, cfg: dict, recorded_scores: dict[str, float]):
    print(f"Plotting {task_name}...")
    mod = load_module(SOL_DIR / cfg["solution"])
    smiles_list = getattr(mod, cfg["func"])()

    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    mols = [m for m in mols if m is not None]
    if not mols:
        print(f"  No valid molecules for {task_name}, skipping")
        return

    props = compute_properties(mols)
    best_score = recorded_scores.get(task_name, 0.0)
    prop_names = cfg["props"]
    n_props = len(prop_names)
    thresholds = PROP_THRESHOLDS.get(task_name, {})
    colors = [ACCENT_BLUE, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_PURPLE, "#E5534B"]

    # Render the full molecule grid
    grid_img = mol_grid_to_array(mols, molsPerRow=10, subImgSize=(200, 150))

    # Figure: big grid on top, small histogram row on bottom
    grid_h, grid_w = grid_img.shape[:2]
    hist_height = 2.0  # inches
    fig_w = 14
    grid_fig_h = fig_w * grid_h / grid_w
    fig_h = grid_fig_h + hist_height + 0.8  # padding for title + gap

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(DARK_BG)

    gs = gridspec.GridSpec(2, 1, figure=fig,
                           height_ratios=[grid_fig_h, hist_height],
                           hspace=0.15)

    # ── Top: molecule grid ──
    ax_grid = fig.add_subplot(gs[0])
    ax_grid.set_facecolor(DARK_BG)
    ax_grid.axis("off")
    ax_grid.imshow(grid_img)

    # ── Bottom: property histograms ──
    inner_gs = gridspec.GridSpecFromSubplotSpec(1, n_props, subplot_spec=gs[1], wspace=0.35)

    for pi, pname in enumerate(prop_names):
        ax = fig.add_subplot(inner_gs[0, pi])
        style_ax(ax)
        vals = props[pname]
        color = colors[pi % len(colors)]
        ax.hist(vals, bins=20, color=color, alpha=0.75, edgecolor=DARK_BG, linewidth=0.5)
        ax.set_xlabel(pname, fontsize=9)
        if pi == 0:
            ax.set_ylabel("Count", fontsize=9)

        # Threshold lines
        for thresh_val, label in thresholds.get(pname, []):
            ax.axvline(thresh_val, color="white", linewidth=1.2, linestyle="--", alpha=0.5)

        # Median line + label
        med = statistics.median(vals)
        ax.axvline(med, color=color, linewidth=1.5, linestyle="-", alpha=0.9)
        ax.text(0.95, 0.92, f"med={med:.2f}", transform=ax.transAxes,
                fontsize=7, color=TEXT_COLOR, ha="right", va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=PANEL_BG,
                          edgecolor=BORDER_COLOR, alpha=0.8))

    title = f"MolOpt: {task_name.replace('_', ' ').title()}"
    fig.suptitle(title, fontsize=14, fontweight="bold", color=TEXT_COLOR, y=0.995)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"solution_{task_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"  Saved {out_path}")
    plt.close(fig)


def main():
    scores = load_recorded_scores()
    for task_name, cfg in TASKS.items():
        sol_path = SOL_DIR / cfg["solution"]
        if not sol_path.exists():
            print(f"Skipping {task_name}: {sol_path} not found")
            continue
        plot_task(task_name, cfg, scores)
    print("Done.")


if __name__ == "__main__":
    main()
