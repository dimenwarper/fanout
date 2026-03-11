#!/usr/bin/env python3
"""Visualize winning MolOpt solutions from the diverse_test run.

Generates multi-panel PNGs for each task showing molecule grids, property
distributions, Tanimoto similarity heatmaps, Murcko scaffolds, and (for
combinatorial solutions) building block diagrams.

Usage:
    uv run python benchmarks/molopt/plot_solutions.py
"""

from __future__ import annotations

import ast
import importlib.util
import json
import statistics
from collections import Counter
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, QED as QEDModule
from rdkit.Chem import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles

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

TASK_COLORS = {
    "maximize_qed": ACCENT_BLUE,
    "qed_logp_balance": ACCENT_ORANGE,
    "constrained_generation": ACCENT_GREEN,
    "drug_candidate": ACCENT_PURPLE,
}

TASKS = {
    "maximize_qed": {
        "func": "maximize_qed",
        "solution": "maximize_qed_1.py",
        "combinatorial": False,
        "props": ["QED", "LogP", "MW"],
        "benchmark": 0.9,
    },
    "qed_logp_balance": {
        "func": "qed_logp_balance",
        "solution": "qed_logp_balance_1.py",
        "combinatorial": False,
        "props": ["QED", "LogP", "MW"],
        "benchmark": 0.85,
    },
    "constrained_generation": {
        "func": "constrained_generation",
        "solution": "constrained_generation_1.py",
        "combinatorial": True,
        "props": ["QED", "LogP", "MW", "TPSA", "Rings"],
        "benchmark": 0.85,
        "fragments": {"Linkers": "Linkers", "L": "L_list", "R": "R_list"},
    },
    "drug_candidate": {
        "func": "drug_candidate",
        "solution": "drug_candidate_1.py",
        "combinatorial": True,
        "props": ["QED", "LogP", "MW", "HBD", "HBA"],
        "benchmark": 0.85,
        "fragments": {"R1": "R1", "L": "L", "R2": "R2"},
    },
}

# Property thresholds for vertical reference lines
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


def extract_local_lists(source_path: Path, func_name: str) -> dict[str, list[str]]:
    """Extract local list-of-strings assignments from a function via AST."""
    source = source_path.read_text()
    tree = ast.parse(source)
    result = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    target = stmt.targets[0]
                    if isinstance(target, ast.Name) and isinstance(stmt.value, ast.List):
                        items = []
                        for elt in stmt.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                items.append(elt.value)
                        if items:
                            result[target.id] = items
    return result


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


def compute_similarity_matrix(mols: list) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(m) for m in mols]
    n = len(fps)
    sim = np.zeros((n, n))
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        sim[i] = sims
    return sim


def mol_grid_to_array(mols, legends=None, molsPerRow=4, subImgSize=(300, 300)):
    """Render molecule grid to numpy array."""
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=molsPerRow,
        subImgSize=subImgSize,
        legends=legends or [None] * len(mols),
        useSVG=False,
    )
    if isinstance(img, Image.Image):
        return np.array(img)
    # If it returned bytes/PNG
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return np.array(Image.open(buf))


def parse_fragment(smi: str):
    """Try to parse a fragment SMILES, handling wildcard atoms and templates."""
    # Replace format placeholders {} with a methyl group for visualization
    cleaned = smi.replace("{}", "C")
    mol = Chem.MolFromSmiles(cleaned)
    if mol is not None:
        return mol
    # Try with wildcard replacement
    cleaned2 = smi.replace("{}", "[H]").replace("*", "[H]")
    mol = Chem.MolFromSmiles(cleaned2)
    if mol is not None:
        return mol
    # Try original as-is
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return mol
    # Try as SMARTS
    mol = Chem.MolFromSmarts(cleaned)
    return mol


def extract_scaffolds(smiles_list: list[str]) -> list[tuple[str, int]]:
    """Extract Murcko scaffolds and return (scaffold_smi, count) sorted by count."""
    scaffolds = []
    for smi in smiles_list:
        try:
            sc = MurckoScaffoldSmilesFromSmiles(smi)
            if sc:
                scaffolds.append(sc)
        except Exception:
            continue
    counts = Counter(scaffolds)
    return counts.most_common()


# ── Plotting ──────────────────────────────────────────

def plot_task(task_name: str, cfg: dict, recorded_scores: dict[str, float]):
    print(f"Plotting {task_name}...")
    mod = load_module(SOL_DIR / cfg["solution"])
    func = getattr(mod, cfg["func"])
    smiles_list = func()

    mols = []
    valid_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            valid_smiles.append(smi)

    if not mols:
        print(f"  No valid molecules for {task_name}, skipping")
        return

    n_mols = len(mols)
    props = compute_properties(mols)
    sim_matrix = compute_similarity_matrix(mols)
    scaffolds = extract_scaffolds(valid_smiles)
    best_score = recorded_scores.get(task_name, 0.0)
    accent = TASK_COLORS[task_name]

    # Determine layout: 4 rows for non-combinatorial, 5 for combinatorial
    is_combo = cfg["combinatorial"]
    n_prop_cols = len(cfg["props"])
    n_rows = 5 if is_combo else 4
    fig_height = 7 * n_rows / 4

    fig = plt.figure(figsize=(16, fig_height))
    fig.patch.set_facecolor(DARK_BG)

    # Height ratios
    if is_combo:
        gs = gridspec.GridSpec(n_rows, 2, figure=fig,
                               height_ratios=[1.3, 1.0, 1.0, 1.0, 1.0],
                               hspace=0.35, wspace=0.3)
    else:
        gs = gridspec.GridSpec(n_rows, 2, figure=fig,
                               height_ratios=[1.3, 1.0, 1.0, 1.0],
                               hspace=0.35, wspace=0.3)

    # ── Row 0: Molecule grid (spans full width) ──
    ax_grid = fig.add_subplot(gs[0, :])
    style_ax(ax_grid)
    ax_grid.axis("off")

    indices = np.linspace(0, n_mols - 1, min(16, n_mols), dtype=int)
    sample_mols = [mols[i] for i in indices]
    sample_legends = [f"#{i}" for i in indices]

    grid_img = mol_grid_to_array(sample_mols, legends=sample_legends,
                                  molsPerRow=8, subImgSize=(250, 200))
    ax_grid.imshow(grid_img)
    ax_grid.set_title(f"Representative Molecules ({len(sample_mols)} of {n_mols})",
                      fontsize=12, fontweight="bold", pad=8)

    # ── Row 1: Property distributions (left) + Similarity heatmap (right) ──
    # Property distributions
    ax_props = fig.add_subplot(gs[1, 0])
    style_ax(ax_props)

    prop_names = cfg["props"]
    thresholds = PROP_THRESHOLDS.get(task_name, {})
    colors = [ACCENT_BLUE, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_PURPLE, "#E5534B"]

    # Use subplots within this area via inset axes
    prop_axes = []
    n_p = len(prop_names)
    inner_gs = gridspec.GridSpecFromSubplotSpec(1, n_p, subplot_spec=gs[1, 0], wspace=0.4)
    # Remove the parent ax_props
    ax_props.remove()

    for pi, pname in enumerate(prop_names):
        ax_p = fig.add_subplot(inner_gs[0, pi])
        style_ax(ax_p)
        vals = props[pname]
        color = colors[pi % len(colors)]
        ax_p.hist(vals, bins=20, color=color, alpha=0.7, edgecolor=DARK_BG, linewidth=0.5)
        ax_p.set_xlabel(pname, fontsize=8)
        if pi == 0:
            ax_p.set_ylabel("Count", fontsize=8)

        # Draw threshold lines
        for thresh_val, label in thresholds.get(pname, []):
            ax_p.axvline(thresh_val, color=ACCENT_ORANGE if "min" in label or "low" in label else "#E5534B",
                         linewidth=1.5, linestyle="--", alpha=0.8)

        # Add median annotation
        med = statistics.median(vals)
        ax_p.axvline(med, color="white", linewidth=1, linestyle=":", alpha=0.6)
        ax_p.text(0.95, 0.95, f"med={med:.2f}", transform=ax_p.transAxes,
                  fontsize=7, color=TEXT_COLOR, ha="right", va="top",
                  fontfamily="monospace",
                  bbox=dict(boxstyle="round,pad=0.2", facecolor=PANEL_BG, edgecolor=BORDER_COLOR, alpha=0.8))
        prop_axes.append(ax_p)

    # Similarity heatmap
    ax_sim = fig.add_subplot(gs[1, 1])
    style_ax(ax_sim)
    im = ax_sim.imshow(sim_matrix, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    cb = fig.colorbar(im, ax=ax_sim, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    cb.outline.set_edgecolor(BORDER_COLOR)
    cb.set_label("Tanimoto Similarity", color=TEXT_COLOR, fontsize=8)

    # Count violations
    n_pairs = n_mols * (n_mols - 1) // 2
    violations = np.sum(np.triu(sim_matrix, k=1) >= 0.6)
    viol_frac = violations / n_pairs if n_pairs > 0 else 0
    div_mult = 1.0 - viol_frac

    ax_sim.set_title(f"Tanimoto Similarity ({violations} violations, div={div_mult:.3f})",
                     fontsize=10, fontweight="bold", pad=6)
    ax_sim.set_xlabel("Molecule", fontsize=8)
    ax_sim.set_ylabel("Molecule", fontsize=8)

    # ── Row 2: Scaffold gallery (spans full width) ──
    ax_scaff = fig.add_subplot(gs[2, :])
    style_ax(ax_scaff)
    ax_scaff.axis("off")

    top_scaffolds = scaffolds[:12]
    if top_scaffolds:
        sc_mols = []
        sc_labels = []
        for sc_smi, count in top_scaffolds:
            mol = Chem.MolFromSmiles(sc_smi)
            if mol is not None:
                sc_mols.append(mol)
                sc_labels.append(f"n={count}")

        if sc_mols:
            sc_img = mol_grid_to_array(sc_mols, legends=sc_labels,
                                        molsPerRow=min(len(sc_mols), 6),
                                        subImgSize=(220, 180))
            ax_scaff.imshow(sc_img)
    ax_scaff.set_title(f"Murcko Scaffolds (top {len(top_scaffolds)} of {len(scaffolds)} unique)",
                       fontsize=11, fontweight="bold", pad=8)

    # ── Row 3: Stats summary (spans full width) ──
    ax_stats = fig.add_subplot(gs[3, :])
    style_ax(ax_stats)
    ax_stats.axis("off")

    stats_lines = [
        f"Task: {task_name}  |  Score: {best_score:.4f}  |  Benchmark: {cfg['benchmark']}",
        f"Molecules: {n_mols}  |  Unique scaffolds: {len(scaffolds)}  |  Diversity: {div_mult:.4f}",
        "",
    ]
    for pname in prop_names:
        vals = props[pname]
        stats_lines.append(
            f"  {pname:>6s}: median={statistics.median(vals):.2f}  "
            f"mean={np.mean(vals):.2f}  min={min(vals):.2f}  max={max(vals):.2f}"
        )

    ax_stats.text(0.05, 0.9, "\n".join(stats_lines), transform=ax_stats.transAxes,
                  fontsize=10, color=TEXT_COLOR, fontfamily="monospace",
                  verticalalignment="top", linespacing=1.6,
                  bbox=dict(boxstyle="round,pad=0.8", facecolor=PANEL_BG, edgecolor=BORDER_COLOR))

    # ── Row 4 (combinatorial only): Building blocks ──
    if is_combo:
        frag_config = cfg["fragments"]
        frag_names = list(frag_config.keys())

        # Extract local variables from the function source via AST
        local_lists = extract_local_lists(SOL_DIR / cfg["solution"], cfg["func"])

        inner_gs2 = gridspec.GridSpecFromSubplotSpec(1, len(frag_names),
                                                      subplot_spec=gs[4, :], wspace=0.3)

        for fi, fname in enumerate(frag_names):
            ax_frag = fig.add_subplot(inner_gs2[0, fi])
            style_ax(ax_frag)
            ax_frag.axis("off")

            var_name = frag_config[fname]
            frag_list = local_lists.get(var_name)
            if frag_list is None:
                ax_frag.set_title(f"{fname}: not found", fontsize=10, fontweight="bold")
                continue

            # Parse fragments
            frag_mols = []
            frag_labels = []
            for i, frag_smi in enumerate(frag_list[:12]):
                mol = parse_fragment(frag_smi)
                if mol is not None:
                    frag_mols.append(mol)
                    frag_labels.append(f"{fname}[{i}]")

            if frag_mols:
                per_row = min(len(frag_mols), 4)
                frag_img = mol_grid_to_array(frag_mols, legends=frag_labels,
                                              molsPerRow=per_row,
                                              subImgSize=(180, 150))
                ax_frag.imshow(frag_img)
            ax_frag.set_title(f"{fname} ({len(frag_list)} fragments)",
                              fontsize=10, fontweight="bold", pad=6)

    # Title
    title = f"MolOpt: {task_name.replace('_', ' ').title()}  |  Score = {best_score:.4f}"
    fig.suptitle(title, fontsize=15, fontweight="bold", color=TEXT_COLOR, y=1.01)

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
