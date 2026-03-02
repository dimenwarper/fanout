#!/usr/bin/env python3
"""Retro terminal animation demo for the fanout project."""

import sys
import time
import signal
import shutil

# ── ANSI escape helpers ──────────────────────────────────────────────────────

ESC = "\033["
CLEAR = f"{ESC}2J{ESC}H"
HIDE_CURSOR = f"{ESC}?25l"
SHOW_CURSOR = f"{ESC}?25h"
BOLD = f"{ESC}1m"
DIM = f"{ESC}2m"
ITALIC = f"{ESC}3m"
STRIKE = f"{ESC}9m"
RESET = f"{ESC}0m"
BLINK = f"{ESC}5m"

# 16 colors
BLACK = f"{ESC}30m"
RED = f"{ESC}31m"
GREEN = f"{ESC}32m"
YELLOW = f"{ESC}33m"
BLUE = f"{ESC}34m"
MAGENTA = f"{ESC}35m"
CYAN = f"{ESC}36m"
WHITE = f"{ESC}37m"
GRAY = f"{ESC}90m"
BRIGHT_RED = f"{ESC}91m"
BRIGHT_GREEN = f"{ESC}92m"
BRIGHT_YELLOW = f"{ESC}93m"
BRIGHT_CYAN = f"{ESC}96m"
BRIGHT_WHITE = f"{ESC}97m"

BG_RED = f"{ESC}41m"
BG_BLACK = f"{ESC}40m"

W = 80  # terminal width


def cleanup(*_):
    sys.stdout.write(SHOW_CURSOR + RESET)
    sys.stdout.flush()
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)


def clear():
    sys.stdout.write(CLEAR)
    sys.stdout.flush()


def goto(row, col):
    sys.stdout.write(f"{ESC}{row};{col}H")
    sys.stdout.flush()


def put(row, col, text):
    goto(row, col)
    sys.stdout.write(text)
    sys.stdout.flush()


def center(text, width=W):
    """Center plain text (no ANSI) in given width."""
    return text.center(width)


def center_col(text_len, width=W):
    """Return column to center text of given length."""
    return max(1, (width - text_len) // 2 + 1)


def typewrite(row, col, text, delay=0.03, color=""):
    """Type text character by character."""
    goto(row, col)
    for ch in text:
        sys.stdout.write(f"{color}{ch}{RESET}")
        sys.stdout.flush()
        time.sleep(delay)


def typewrite_centered(row, text, delay=0.03, color=""):
    col = center_col(len(text))
    typewrite(row, col, text, delay, color)


def pause(seconds=1.5):
    time.sleep(seconds)


def draw_box(top, left, width, height, double=True):
    """Draw a box with Unicode box-drawing characters."""
    if double:
        tl, tr, bl, br, h, v = "╔", "╗", "╚", "╝", "═", "║"
    else:
        tl, tr, bl, br, h, v = "┌", "┐", "└", "┘", "─", "│"
    put(top, left, tl + h * (width - 2) + tr)
    for r in range(top + 1, top + height - 1):
        put(r, left, v)
        put(r, left + width - 1, v)
    put(top + height - 1, left, bl + h * (width - 2) + br)


# ── Scene 1: Title ──────────────────────────────────────────────────────────

TITLE_ART = [
    "███████╗ █████╗ ███╗   ██╗ ██████╗ ██╗   ██╗████████╗",
    "██╔════╝██╔══██╗████╗  ██║██╔═══██╗██║   ██║╚══██╔══╝",
    "█████╗  ███████║██╔██╗ ██║██║   ██║██║   ██║   ██║   ",
    "██╔══╝  ██╔══██║██║╚██╗██║██║   ██║██║   ██║   ██║   ",
    "██║     ██║  ██║██║ ╚████║╚██████╔╝╚██████╔╝   ██║   ",
    "╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝    ╚═╝   ",
]


def scene_title():
    clear()
    draw_box(1, 1, W, 24, double=True)

    # Draw title art line by line with animation
    art_width = len(TITLE_ART[0])
    start_col = center_col(art_width)
    start_row = 5

    for i, line in enumerate(TITLE_ART):
        put(start_row + i, start_col, f"{BOLD}{MAGENTA}{line}{RESET}")
        time.sleep(0.08)

    pause(0.5)

    # Subtitle typed out
    subtitle = "Sample.  Evaluate.  Evolve."
    typewrite_centered(14, subtitle, delay=0.04, color=f"{DIM}{WHITE}")

    pause(0.4)

    # Tagline
    tagline = "Evolutionary optimization with competing LLMs"
    typewrite_centered(16, tagline, delay=0.025, color=f"{GRAY}")

    # Version / decorative line
    deco = "· · ·"
    put(18, center_col(len(deco)), f"{GRAY}{deco}{RESET}")

    pause(2.5)


# ── Scene 2: What is it ─────────────────────────────────────────────────────

def scene_what():
    clear()
    draw_box(1, 1, W, 24, double=True)

    put(3, center_col(12), f"{BOLD}{CYAN}HOW IT WORKS{RESET}")
    put(4, center_col(12), f"{CYAN}────────────{RESET}")

    lines = [
        (7,  f"{WHITE}Multiple LLMs compete to solve {BRIGHT_CYAN}hard problems{WHITE}.{RESET}"),
        (9,  f"{WHITE}The {BRIGHT_CYAN}best solutions{WHITE} survive and {BRIGHT_CYAN}improve{WHITE}.{RESET}"),
        (11, f"{WHITE}Like evolution — but with {BRIGHT_CYAN}neural networks{WHITE}.{RESET}"),
    ]

    for row, text in lines:
        # Strip ANSI for centering calculation
        plain = text
        for code in [WHITE, BRIGHT_CYAN, CYAN, RESET, BOLD]:
            plain = plain.replace(code, "")
        col = center_col(len(plain))
        goto(row, col)
        # Typewrite with embedded colors
        for ch in text:
            sys.stdout.write(ch)
            sys.stdout.flush()
            # Only delay on visible characters
            if ch not in ("\033",) and not (len(ch) == 1 and ord(ch) < 32):
                time.sleep(0.02)

    pause(0.8)

    # Strategy diagram
    put(14, center_col(50), f"{GRAY}┌─────────┐    ┌──────────┐    ┌─────────┐{RESET}")
    time.sleep(0.1)
    put(15, center_col(50), f"{GRAY}│{YELLOW} Sample  {GRAY}│───▶│{CYAN} Evaluate {GRAY}│───▶│{GREEN} Evolve  {GRAY}│{RESET}")
    time.sleep(0.1)
    put(16, center_col(50), f"{GRAY}└─────────┘    └──────────┘    └─────────┘{RESET}")
    time.sleep(0.1)
    put(17, center_col(50), f"{GRAY}  GPT, Claude      Score         Keep the   {RESET}")
    put(18, center_col(50), f"{GRAY}  Gemini, ...    correctness      winners    {RESET}")

    pause(3.0)


# ── Scene 3: Benchmark showcase ─────────────────────────────────────────────

BENCHMARKS = [
    ("CodeEvolve",       "Algorithm Discovery",    "Circle packing, kissing numbers"),
    ("KernelBench",      "CUDA Optimization",      "GPU kernel speedups"),
    ("MiniF2F",          "Theorem Proving",         "Formal proofs in Lean 4"),
    ("MolOpt",           "Molecular Optimization",  "Drug-like molecule design"),
    ("MNIST-Weights",    "Neural Net Weights",      "Raw weight generation"),
    ("CIFAR-10",         "CNN Weight Generation",   "Vision model weights"),
]


def scene_benchmarks():
    clear()
    draw_box(1, 1, W, 24, double=True)

    put(3, center_col(10), f"{BOLD}{YELLOW}BENCHMARKS{RESET}")
    put(4, center_col(10), f"{YELLOW}══════════{RESET}")

    # Table header
    hdr_row = 6
    put(hdr_row, 5, f"{BOLD}{WHITE}  Name              Domain                   Challenge{RESET}")
    put(hdr_row + 1, 5, f"{GRAY}  {'─' * 68}{RESET}")

    # Animate rows
    for i, (name, domain, desc) in enumerate(BENCHMARKS):
        row = hdr_row + 2 + i * 2
        # Name in cyan, rest in white
        line = f"  {BRIGHT_CYAN}{name:<18}{WHITE}{domain:<25}{GRAY}{desc}{RESET}"
        put(row, 5, line)
        time.sleep(0.35)

    pause(3.0)


# ── Scene 4: MolOpt spotlight ────────────────────────────────────────────────

SMILES_FRAGMENTS = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",
    "CC(C)Cc1ccc(CC(=O)O)cc1",
    "CC12CCC3C(CCC4CC(=O)CCC43C)C1CCC2O",
    "c1ccc(-c2nnc(-c3ccccc3)o2)cc1",
    "NS(=O)(=O)c1cc2c(cc1Cl)NCNS2(=O)=O",
    "CC(=O)Nc1ccc(O)cc1",
    "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
]


def scene_molopt():
    clear()
    draw_box(1, 1, W, 24, double=True)

    put(3, center_col(16), f"{BOLD}{MAGENTA}MOLOPT SPOTLIGHT{RESET}")
    put(4, center_col(16), f"{MAGENTA}════════════════{RESET}")

    typewrite_centered(6, "Design 100 diverse drug-like molecules.", delay=0.025, color=WHITE)
    typewrite_centered(7, "Median QED score. No cheating allowed.", delay=0.025, color=GRAY)

    pause(1.0)

    # Score ticking up
    put(9, center_col(30), f"{WHITE}Median QED Score:{RESET}")
    scores = [0.00, 0.15, 0.34, 0.52, 0.69, 0.77, 0.81, 0.87, 0.93]
    for score in scores:
        if score >= 0.85:
            color = BRIGHT_GREEN
        elif score >= 0.6:
            color = BRIGHT_YELLOW
        else:
            color = WHITE
        put(10, center_col(8), f"{BOLD}{color}  {score:.2f}  {RESET}")
        time.sleep(0.3)

    pause(0.5)

    # SMILES scrolling
    put(12, center_col(20), f"{GRAY}Generated SMILES:{RESET}")
    smiles_row = 14
    for i, smi in enumerate(SMILES_FRAGMENTS):
        display = smi[:60]
        # Clear previous line
        put(smiles_row, 8, " " * 64)
        put(smiles_row, 8, f"{GREEN}{display}{RESET}")
        time.sleep(0.4)

    pause(0.5)

    # Task scores
    put(16, center_col(44), f"{GRAY}────────────────────────────────────────────{RESET}")
    tasks = [
        ("maximize_qed",         0.93),
        ("qed_logp_balance",     0.86),
        ("constrained_gen",      0.99),
        ("drug_candidate",       1.00),
    ]
    for i, (task, score) in enumerate(tasks):
        if score >= 0.95:
            c = BRIGHT_GREEN
        elif score >= 0.80:
            c = BRIGHT_YELLOW
        else:
            c = WHITE
        put(17 + i, 18, f"{WHITE}{task:<22}{c}{BOLD}{score:.2f}{RESET}")
        time.sleep(0.3)

    pause(2.5)


# ── Scene 5: Anti-cheat ─────────────────────────────────────────────────────

def scene_anticheat():
    clear()
    draw_box(1, 1, W, 24, double=True)

    pause(0.5)

    # Flash warning
    for _ in range(3):
        put(6, center_col(22), f"{BOLD}{BG_RED}{BRIGHT_WHITE}  CHEATING DETECTED!  {RESET}")
        time.sleep(0.25)
        put(6, center_col(22), " " * 22)
        time.sleep(0.15)
    put(6, center_col(22), f"{BOLD}{BG_RED}{BRIGHT_WHITE}  CHEATING DETECTED!  {RESET}")

    pause(1.0)

    # Struck-through import
    put(9, center_col(40), f"{RED}{STRIKE}from rdkit import Chem{RESET}")
    time.sleep(0.5)
    put(10, center_col(40), f"{RED}{STRIKE}from sklearn import svm{RESET}")
    time.sleep(0.5)
    put(11, center_col(40), f"{RED}{STRIKE}import tensorflow as tf{RESET}")

    pause(1.2)

    # Green resolution
    typewrite_centered(14, "No ML libraries. No cheminformatics.", delay=0.03, color=f"{BRIGHT_GREEN}")
    typewrite_centered(15, "LLMs must use pure domain knowledge.", delay=0.03, color=f"{BRIGHT_GREEN}")

    pause(0.8)

    # Enforcement details
    put(17, center_col(46), f"{GRAY}Static analysis  ·  1s timeout  ·  Score → 0.0{RESET}")

    # Banned items scrolling
    banned = ["rdkit", "openbabel", "sklearn", "torch", "tensorflow", "keras", "jax", "__import__()"]
    goto(19, center_col(50))
    sys.stdout.write(f"{DIM}{RED}")
    for item in banned:
        sys.stdout.write(f" ✗ {item} ")
        sys.stdout.flush()
        time.sleep(0.15)
    sys.stdout.write(RESET)

    pause(2.5)


# ── Scene 6: Results ─────────────────────────────────────────────────────────

RESULTS = [
    ("CodeEvolve",    "circle_packing",     2.636,  "2.636",  "ratio"),
    ("CodeEvolve",    "kissing_number",     582,    "582",    "count"),
    ("CodeEvolve",    "autocorrelation",    0.663,  "0.663",  "score"),
    ("MolOpt",        "maximize_qed",       0.930,  "0.930",  "score"),
    ("MolOpt",        "drug_candidate",     1.000,  "1.000",  "score"),
    ("MNIST-Wts",     "classify_all",       0.987,  "98.7%",  "acc"),
]


def scene_results():
    clear()
    draw_box(1, 1, W, 24, double=True)

    put(3, center_col(20), f"{BOLD}{GREEN}RESULTS: DIVERSE RUN{RESET}")
    put(4, center_col(20), f"{GREEN}════════════════════{RESET}")

    # Table
    put(6, 6, f"{BOLD}{WHITE} Benchmark        Task                 Score{RESET}")
    put(7, 6, f"{GRAY} {'─' * 62}{RESET}")

    for i, (bench, task, val, display, _kind) in enumerate(RESULTS):
        row = 8 + i * 2
        if val >= 0.95 or val >= 100:
            c = BRIGHT_GREEN
        elif val >= 0.80 or val >= 500:
            c = BRIGHT_YELLOW
        else:
            c = WHITE

        put(row, 6, f" {CYAN}{bench:<17}{WHITE}{task:<22}{c}{BOLD}{display:>8}{RESET}")
        time.sleep(0.4)

    pause(1.0)

    # Model roster
    put(21, center_col(60), f"{GRAY}Models: GPT-4o · Claude Opus · Gemini Pro · Kimi K2 · GLM-4 ·{RESET}")
    put(22, center_col(30), f"{GRAY}GPT-5.2 Codex · Claude Sonnet{RESET}")

    pause(3.0)


# ── Scene 7: Closing ─────────────────────────────────────────────────────────

def scene_closing():
    clear()
    draw_box(1, 1, W, 24, double=True)

    put(8, center_col(len(TITLE_ART[0])), f"{DIM}{MAGENTA}{TITLE_ART[0]}{RESET}")
    put(9, center_col(len(TITLE_ART[1])), f"{DIM}{MAGENTA}{TITLE_ART[1]}{RESET}")
    put(10, center_col(len(TITLE_ART[2])), f"{DIM}{MAGENTA}{TITLE_ART[2]}{RESET}")
    put(11, center_col(len(TITLE_ART[3])), f"{DIM}{MAGENTA}{TITLE_ART[3]}{RESET}")
    put(12, center_col(len(TITLE_ART[4])), f"{DIM}{MAGENTA}{TITLE_ART[4]}{RESET}")
    put(13, center_col(len(TITLE_ART[5])), f"{DIM}{MAGENTA}{TITLE_ART[5]}{RESET}")

    pause(0.5)

    url = "github.com/dimenwarper/fanout"
    typewrite_centered(16, url, delay=0.04, color=f"{BOLD}{BRIGHT_WHITE}")

    pause(0.8)

    put(18, center_col(38), f"{GRAY}Sample. Evaluate. Evolve. Repeat.{RESET}")

    # Blinking cursor
    cursor_col = center_col(len(url)) + len(url) + 1
    for _ in range(8):
        put(16, cursor_col, f"{BRIGHT_WHITE}█{RESET}")
        time.sleep(0.5)
        put(16, cursor_col, " ")
        time.sleep(0.5)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    sys.stdout.write(HIDE_CURSOR)
    sys.stdout.flush()

    try:
        scene_title()
        scene_what()
        scene_benchmarks()
        scene_molopt()
        scene_anticheat()
        scene_results()
        scene_closing()
    finally:
        sys.stdout.write(SHOW_CURSOR + RESET + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
