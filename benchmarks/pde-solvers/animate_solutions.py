#!/usr/bin/env python3
"""Animate PDE reference vs fanout solutions.

Creates GIF animations showing oracle (reference) solutions alongside the best
fanout-evolved solutions, evolving over time.

- Burgers 1D & KS 1D: overlaid line plots (reference vs fanout)
- Navier-Stokes 2D: side-by-side vorticity heatmaps

Usage:
    python benchmarks/pde-solvers/animate_solutions.py
    python benchmarks/pde-solvers/animate_solutions.py --instance 5
    python benchmarks/pde-solvers/animate_solutions.py --tasks burgers_1d ks_1d
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
REF_DIR = BENCH_DIR / "reference"
SOL_DIR = BENCH_DIR / "runs" / "diverse_full" / "solutions"
OUT_DIR = BENCH_DIR / "plots"

TASKS = {
    "burgers_1d": {
        "label": "Burgers 1D",
        "domain": (0, 2 * np.pi),
        "domain_label": "x",
        "solution_file": "burgers_1d_1.py",
        "extra_args": {"nu": 0.01},
        "type": "1d",
    },
    "navier_stokes_2d": {
        "label": "Navier-Stokes 2D",
        "domain": (0, 2 * np.pi),
        "domain_label": "x, y",
        "solution_file": "navier_stokes_2d_1.py",
        "extra_args": {"nu": 1e-3},
        "type": "2d",
    },
    "ks_1d": {
        "label": "Kuramoto-Sivashinsky 1D",
        "domain": (0, 64 * np.pi),
        "domain_label": "x",
        "solution_file": "ks_1d_1.py",
        "extra_args": {},
        "type": "1d",
    },
}


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("sol", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_reference(task_name: str, instance: int):
    ic = np.load(REF_DIR / f"{task_name}_ic_{instance:02d}.npy")
    ref = np.load(REF_DIR / f"{task_name}_ref_{instance:02d}.npy")
    t_coords = np.load(REF_DIR / f"{task_name}_t_coordinates.npy")
    return ic, ref, t_coords


def compute_solution(task_name: str, ic: np.ndarray, t_coords: np.ndarray):
    info = TASKS[task_name]
    sol_path = SOL_DIR / info["solution_file"]
    mod = load_module(sol_path)

    # Batch dimension
    u0_batch = ic[np.newaxis]
    result = mod.solve_pde(u0_batch, t_coords, **info["extra_args"])
    return result[0]  # remove batch dim -> [T, *spatial]


def animate_1d(task_name: str, instance: int = 3, fps: int = 8):
    info = TASKS[task_name]
    ic, ref, t_coords = load_reference(task_name, instance)
    print(f"Computing {info['label']} solution (instance {instance})...")
    pred = compute_solution(task_name, ic, t_coords)

    # Prepend IC as t=0 frame
    ref_full = np.vstack([ic[np.newaxis], ref])   # [T+1, nx]
    pred_full = np.vstack([ic[np.newaxis], pred])  # [T+1, nx]
    t_full = np.concatenate([[0.0], t_coords])

    nx = ic.shape[0]
    x = np.linspace(info["domain"][0], info["domain"][1], nx, endpoint=False)

    # Compute nRMSE per frame
    nrmse_per_frame = []
    for i in range(len(t_full)):
        ref_norm = np.linalg.norm(ref_full[i])
        if ref_norm > 1e-12:
            nrmse_per_frame.append(np.linalg.norm(pred_full[i] - ref_full[i]) / ref_norm)
        else:
            nrmse_per_frame.append(0.0)

    fig, (ax_main, ax_err) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[3, 1],
                                           gridspec_kw={"hspace": 0.35})
    fig.patch.set_facecolor("#0D1117")
    for ax in (ax_main, ax_err):
        ax.set_facecolor("#161B22")
        ax.tick_params(colors="#C9D1D9")
        ax.xaxis.label.set_color("#C9D1D9")
        ax.yaxis.label.set_color("#C9D1D9")
        for spine in ax.spines.values():
            spine.set_color("#30363D")

    ymin = min(ref_full.min(), pred_full.min()) * 1.15
    ymax = max(ref_full.max(), pred_full.max()) * 1.15

    line_ref, = ax_main.plot([], [], color="#58A6FF", linewidth=2, label="Reference", alpha=0.9)
    line_pred, = ax_main.plot([], [], color="#F78166", linewidth=2, label="Fanout", linestyle="--", alpha=0.9)
    fill = ax_main.fill_between(x, 0, 0, color="#F78166", alpha=0.0)  # placeholder
    time_text = ax_main.text(0.02, 0.95, "", transform=ax_main.transAxes,
                              fontsize=12, color="#C9D1D9", verticalalignment="top",
                              fontfamily="monospace")
    nrmse_text = ax_main.text(0.98, 0.95, "", transform=ax_main.transAxes,
                               fontsize=11, color="#F78166", verticalalignment="top",
                               horizontalalignment="right", fontfamily="monospace")

    ax_main.set_xlim(x[0], x[-1])
    ax_main.set_ylim(ymin, ymax)
    ax_main.set_xlabel(info["domain_label"])
    ax_main.set_ylabel("u")
    ax_main.legend(loc="upper right", fontsize=10, facecolor="#161B22", edgecolor="#30363D",
                   labelcolor="#C9D1D9")

    title = ax_main.set_title(f"{info['label']} — Reference vs Fanout (instance {instance})",
                               fontsize=13, color="#C9D1D9", fontweight="bold")

    # Error subplot
    line_err, = ax_err.plot([], [], color="#F78166", linewidth=1.5)
    ax_err.set_xlim(x[0], x[-1])
    err_max = np.max(np.abs(pred_full - ref_full)) * 1.2
    ax_err.set_ylim(-err_max, err_max)
    ax_err.set_xlabel(info["domain_label"])
    ax_err.set_ylabel("Error")
    ax_err.axhline(0, color="#30363D", linewidth=0.5)

    def init():
        line_ref.set_data([], [])
        line_pred.set_data([], [])
        line_err.set_data([], [])
        time_text.set_text("")
        nrmse_text.set_text("")
        return line_ref, line_pred, line_err, time_text, nrmse_text

    def animate_frame(i):
        nonlocal fill
        line_ref.set_data(x, ref_full[i])
        line_pred.set_data(x, pred_full[i])

        # Error fill
        fill.remove()
        err = pred_full[i] - ref_full[i]
        fill = ax_err.fill_between(x, 0, err, color="#F78166", alpha=0.3)
        line_err.set_data(x, err)

        time_text.set_text(f"t = {t_full[i]:.1f}")
        nrmse_text.set_text(f"nRMSE = {nrmse_per_frame[i]:.4f}")
        return line_ref, line_pred, line_err, fill, time_text, nrmse_text

    n_frames = len(t_full)
    # Repeat last frame a few times for pause effect
    frame_indices = list(range(n_frames)) + [n_frames - 1] * 4

    anim = animation.FuncAnimation(fig, animate_frame, init_func=init,
                                    frames=frame_indices, interval=1000 // fps, blit=False)

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"anim_{task_name}.gif"
    anim.save(str(out_path), writer="pillow", fps=fps, dpi=120)
    print(f"Saved {out_path}")
    plt.close(fig)


def animate_2d(task_name: str, instance: int = 3, fps: int = 4):
    info = TASKS[task_name]
    ic, ref, t_coords = load_reference(task_name, instance)
    print(f"Computing {info['label']} solution (instance {instance})...")
    pred = compute_solution(task_name, ic, t_coords)

    ref_full = np.concatenate([ic[np.newaxis], ref], axis=0)
    pred_full = np.concatenate([ic[np.newaxis], pred], axis=0)
    t_full = np.concatenate([[0.0], t_coords])

    vmin = min(ref_full.min(), pred_full.min())
    vmax = max(ref_full.max(), pred_full.max())
    # Symmetric colorbar
    vlim = max(abs(vmin), abs(vmax))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5),
                              gridspec_kw={"width_ratios": [1, 1, 0.08], "wspace": 0.15})
    fig.patch.set_facecolor("#0D1117")
    ax_ref, ax_pred, ax_cb = axes

    for ax in (ax_ref, ax_pred):
        ax.set_facecolor("#161B22")
        ax.tick_params(colors="#C9D1D9")
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_color("#30363D")

    im_ref = ax_ref.imshow(ref_full[0], cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                            origin="lower", extent=[0, 2*np.pi, 0, 2*np.pi])
    im_pred = ax_pred.imshow(pred_full[0], cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                              origin="lower", extent=[0, 2*np.pi, 0, 2*np.pi])

    ax_ref.set_title("Reference", fontsize=12, color="#58A6FF", fontweight="bold")
    ax_pred.set_title("Fanout", fontsize=12, color="#F78166", fontweight="bold")

    cb = fig.colorbar(im_ref, cax=ax_cb)
    cb.ax.tick_params(colors="#C9D1D9")
    cb.outline.set_edgecolor("#30363D")
    cb.set_label("Vorticity (ω)", color="#C9D1D9")

    suptitle = fig.suptitle(f"{info['label']} — instance {instance}  |  t = 0.0",
                             fontsize=13, color="#C9D1D9", fontweight="bold", y=0.98)

    # nRMSE annotation
    nrmse_text = fig.text(0.5, 0.02, "", ha="center", fontsize=11, color="#F78166",
                           fontfamily="monospace")

    def animate_frame(i):
        im_ref.set_data(ref_full[i])
        im_pred.set_data(pred_full[i])

        ref_norm = np.linalg.norm(ref_full[i].ravel())
        if ref_norm > 1e-12:
            nrmse = np.linalg.norm((pred_full[i] - ref_full[i]).ravel()) / ref_norm
        else:
            nrmse = 0.0

        suptitle.set_text(f"{info['label']} — instance {instance}  |  t = {t_full[i]:.1f}")
        nrmse_text.set_text(f"nRMSE = {nrmse:.4f}")
        return im_ref, im_pred, suptitle, nrmse_text

    n_frames = len(t_full)
    frame_indices = list(range(n_frames)) + [n_frames - 1] * 4

    anim = animation.FuncAnimation(fig, animate_frame,
                                    frames=frame_indices, interval=1000 // fps, blit=False)

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"anim_{task_name}.gif"
    anim.save(str(out_path), writer="pillow", fps=fps, dpi=120)
    print(f"Saved {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Animate PDE reference vs fanout solutions")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS.keys()),
                        choices=list(TASKS.keys()), help="Tasks to animate")
    parser.add_argument("--instance", type=int, default=3, help="IC instance index (default: 3)")
    parser.add_argument("--fps", type=int, default=6, help="Frames per second (default: 6)")
    args = parser.parse_args()

    for task_name in args.tasks:
        info = TASKS[task_name]
        if info["type"] == "1d":
            animate_1d(task_name, instance=args.instance, fps=args.fps)
        else:
            animate_2d(task_name, instance=args.instance, fps=args.fps)

    print("Done.")


if __name__ == "__main__":
    main()
