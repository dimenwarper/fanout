#!/usr/bin/env python3
"""Retro terminal animation demo for the fanout project."""

import sys
import time
import signal

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


def typewrite_rich(row, col, text, delay=0.02):
    """Typewrite text that contains embedded ANSI codes."""
    goto(row, col)
    i = 0
    while i < len(text):
        if text[i] == "\033":
            # Consume entire escape sequence
            j = i
            while j < len(text) and text[j] != "m":
                j += 1
            sys.stdout.write(text[i : j + 1])
            i = j + 1
        else:
            sys.stdout.write(text[i])
            sys.stdout.flush()
            time.sleep(delay)
            i += 1
    sys.stdout.write(RESET)
    sys.stdout.flush()


def plain_len(text):
    """Length of text with ANSI codes stripped."""
    import re
    return len(re.sub(r"\033\[[0-9;]*m", "", text))


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

    art_width = len(TITLE_ART[0])
    start_col = center_col(art_width)
    start_row = 5

    for i, line in enumerate(TITLE_ART):
        put(start_row + i, start_col, f"{BOLD}{MAGENTA}{line}{RESET}")
        time.sleep(0.08)

    pause(0.5)

    subtitle = "Sample.  Evaluate.  Evolve."
    typewrite_centered(14, subtitle, delay=0.04, color=f"{DIM}{WHITE}")

    pause(0.4)

    tagline = "Evolutionary optimization with competing LLMs"
    typewrite_centered(16, tagline, delay=0.025, color=f"{GRAY}")

    deco = "· · ·"
    put(18, center_col(len(deco)), f"{GRAY}{deco}{RESET}")

    pause(2.5)


# ── Scene 2: Agents launch ──────────────────────────────────────────────────

def scene_agents():
    clear()
    draw_box(1, 1, W, 24, double=True)

    put(3, center_col(14), f"{BOLD}{CYAN}LAUNCH AGENTS{RESET}")
    put(4, center_col(14), f"{CYAN}──────────────{RESET}")

    typewrite_rich(6, center_col(56),
        f"{WHITE}Fanout spawns {BRIGHT_CYAN}concurrent agents{WHITE} — each backed by a "
        f"different LLM.", delay=0.02)

    pause(0.6)

    # Animated agent spawn
    models = [
        ("Claude Opus",   MAGENTA),
        ("GPT-5.2 Codex", GREEN),
        ("Gemini Pro",    CYAN),
        ("Kimi K2",       YELLOW),
        ("DeepSeek R1",   BLUE),
    ]

    base_row = 9
    for i, (model, color) in enumerate(models):
        col = 12
        # Spawn animation: dot trail then agent box
        for c in range(col, col + 8):
            put(base_row + i, c, f"{DIM}{color}·{RESET}")
            time.sleep(0.02)
        label = f" ▸ {model} "
        put(base_row + i, col + 8, f"{BOLD}{color}{label}{RESET}")
        # Status
        put(base_row + i, col + 8 + len(label) + 2, f"{DIM}{WHITE}spawned{RESET}")
        time.sleep(0.25)

    pause(0.8)

    typewrite_rich(16, center_col(58),
        f"{GRAY}Each agent reads the prompt, studies existing solutions,{RESET}",
        delay=0.02)
    typewrite_rich(17, center_col(52),
        f"{GRAY}writes code, evaluates it — then iterates.{RESET}",
        delay=0.02)

    pause(0.5)

    put(19, center_col(54),
        f"{DIM}{WHITE}Agents are {BRIGHT_CYAN}continuous streams{DIM}{WHITE}"
        f" — not one-shot calls.{RESET}")

    pause(3.0)


# ── Scene 3: The Channel ────────────────────────────────────────────────────

def scene_channel():
    clear()
    draw_box(1, 1, W, 24, double=True)

    put(3, center_col(19), f"{BOLD}{YELLOW}THE SHARED CHANNEL{RESET}")
    put(4, center_col(19), f"{YELLOW}═══════════════════{RESET}")

    typewrite_rich(6, center_col(60),
        f"{WHITE}Agents never talk directly. They communicate through a "
        f"{BRIGHT_YELLOW}Channel{WHITE}.{RESET}", delay=0.02)

    pause(0.8)

    # Channel diagram
    #   Agent ──┐
    #   Agent ──┤── Channel (Redis) ──┤── Solutions
    #   Agent ──┘                     ├── Evaluations
    #                                 └── Memories

    c = 10
    put(9,  c, f"{MAGENTA}  Claude  {GRAY}──┐{RESET}")
    time.sleep(0.2)
    put(10, c, f"{GREEN}  GPT-5   {GRAY}──┤{RESET}")
    time.sleep(0.2)
    put(11, c, f"{CYAN}  Gemini  {GRAY}──┤{RESET}")
    time.sleep(0.2)
    put(12, c, f"{YELLOW}  Kimi    {GRAY}──┘{RESET}")
    time.sleep(0.3)

    # Channel box appears
    draw_box(9, 25, 18, 4, double=False)
    put(10, 27, f"{BOLD}{BRIGHT_YELLOW}Channel{RESET}")
    put(11, 27, f"{DIM}{WHITE}Redis / Memory{RESET}")
    time.sleep(0.4)

    # Arrows to data
    put(9,  43, f"{GRAY}──▸ {BRIGHT_CYAN}Solutions{RESET}")
    time.sleep(0.2)
    put(10, 43, f"{GRAY}──▸ {BRIGHT_CYAN}Evaluations{RESET}")
    time.sleep(0.2)
    put(11, 43, f"{GRAY}──▸ {BRIGHT_CYAN}Memories{RESET}")
    time.sleep(0.2)
    put(12, 43, f"{GRAY}──▸ {BRIGHT_CYAN}Scores{RESET}")

    pause(1.0)

    typewrite_rich(15, center_col(54),
        f"{WHITE}put() / get() / list() — a minimal shared bus.{RESET}",
        delay=0.02)

    typewrite_rich(17, center_col(58),
        f"{GRAY}Every solution, score, and learning flows through here.{RESET}",
        delay=0.02)
    typewrite_rich(18, center_col(56),
        f"{GRAY}Agents read each other's work. Build on what's best.{RESET}",
        delay=0.02)

    pause(3.0)


# ── Scene 4: Strategies ─────────────────────────────────────────────────────

def scene_strategies():
    clear()
    draw_box(1, 1, W, 24, double=True)

    put(3, center_col(21), f"{BOLD}{GREEN}SELECTION STRATEGIES{RESET}")
    put(4, center_col(21), f"{GREEN}═════════════════════{RESET}")

    typewrite_rich(6, center_col(56),
        f"{WHITE}After each round, a {BRIGHT_GREEN}strategy{WHITE}"
        f" picks which solutions survive.{RESET}", delay=0.02)

    pause(0.8)

    strategies = [
        ("Darwinian",
         "Sigmoid-weighted sampling with novelty bonus",
         BRIGHT_GREEN),
        ("Top-K",
         "Keep the K highest-scoring solutions",
         BRIGHT_CYAN),
        ("Epsilon-Greedy",
         "Exploit the best, explore randomly with P(e)",
         BRIGHT_YELLOW),
        ("Pareto",
         "Preserve non-dominated trade-offs across objectives",
         MAGENTA),
    ]

    base_row = 9
    for i, (name, desc, color) in enumerate(strategies):
        row = base_row + i * 3
        put(row, 8, f"{BOLD}{color}▸ {name}{RESET}")
        time.sleep(0.15)
        typewrite(row + 1, 12, desc, delay=0.015, color=f"{DIM}{WHITE}")
        time.sleep(0.2)

    pause(1.0)

    put(22, center_col(52),
        f"{GRAY}Winners become parents. Their code seeds the next round.{RESET}")

    pause(3.0)


# ── Scene 5: The Loop ────────────────────────────────────────────────────────

def scene_loop():
    clear()
    draw_box(1, 1, W, 24, double=True)

    put(3, center_col(22), f"{BOLD}{BRIGHT_CYAN}CONTINUOUS EVOLUTION{RESET}")
    put(4, center_col(22), f"{BRIGHT_CYAN}══════════════════════{RESET}")

    # Animated loop diagram
    steps = [
        ("SAMPLE",   YELLOW,  "Fan out to N models"),
        ("EVALUATE", CYAN,    "Score every solution"),
        ("SELECT",   GREEN,   "Strategy picks survivors"),
        ("REMEMBER", MAGENTA, "Record learnings to memory"),
        ("REFLECT",  BLUE,    "Diagnose failures via LLM"),
        ("EVOLVE",   BRIGHT_YELLOW, "Build next prompt from winners"),
    ]

    base_row = 7
    for i, (step, color, desc) in enumerate(steps):
        row = base_row + i * 2
        # Step number
        put(row, 10, f"{DIM}{WHITE}{i+1}.{RESET}")
        # Arrow animation
        put(row, 14, f"{color}▸{RESET}")
        time.sleep(0.08)
        # Step name
        put(row, 16, f"{BOLD}{color}{step:<10}{RESET}")
        # Description
        put(row, 27, f"{WHITE}{desc}{RESET}")
        time.sleep(0.3)

    pause(0.5)

    # Loop arrow
    put(19, 10, f"{DIM}{GRAY}╰──────── round N+1 ◂── repeat ──────────────╯{RESET}")

    pause(0.8)

    put(21, center_col(58),
        f"{BRIGHT_WHITE}Not a single API call — a {BOLD}living stream{RESET}"
        f"{BRIGHT_WHITE} of collaboration.{RESET}")

    pause(3.5)


# ── Scene 6: Memory & Collaboration ─────────────────────────────────────────

def scene_memory():
    clear()
    draw_box(1, 1, W, 24, double=True)

    put(3, center_col(19), f"{BOLD}{MAGENTA}SHARED MEMORY BANK{RESET}")
    put(4, center_col(19), f"{MAGENTA}═══════════════════{RESET}")

    typewrite_rich(6, center_col(52),
        f"{WHITE}Agents don't just pass code — they share {BRIGHT_MAGENTA}knowledge{WHITE}.{RESET}",
        delay=0.02)

    pause(0.8)

    # Memory types appearing like log entries
    entries = [
        ("hypothesis", "agent-0", "Hexagonal init should beat random for packing"),
        ("learning",   "agent-2", "Sigmoid params > linear for QED scoring"),
        ("strategy",   "agent-1", "Combinatorial fragments guarantee diversity"),
        ("learning",   "agent-3", "Timeout at 0.8s — need simpler loop"),
        ("strategy",   "workflow", "Round 3: top scorer used template approach"),
    ]

    base_row = 9
    colors = {
        "hypothesis": BRIGHT_CYAN,
        "learning": BRIGHT_GREEN,
        "strategy": BRIGHT_YELLOW,
    }

    for i, (mtype, agent, content) in enumerate(entries):
        row = base_row + i * 2
        c = colors.get(mtype, WHITE)
        put(row, 6, f"{DIM}{GRAY}{agent:>10}{RESET}"
                     f" {c}[{mtype}]{RESET}"
                     f" {DIM}{WHITE}{content[:42]}{RESET}")
        time.sleep(0.5)

    pause(0.8)

    typewrite_rich(20, center_col(56),
        f"{GRAY}Memories are synthesized and injected into the next round.{RESET}",
        delay=0.02)
    typewrite_rich(21, center_col(50),
        f"{GRAY}Every agent benefits from every other agent's work.{RESET}",
        delay=0.02)

    pause(3.0)


# ── Scene 7: Closing ─────────────────────────────────────────────────────────

def scene_closing():
    clear()
    draw_box(1, 1, W, 24, double=True)

    for i, line in enumerate(TITLE_ART):
        put(8 + i, center_col(len(line)), f"{DIM}{MAGENTA}{line}{RESET}")

    pause(0.5)

    url = "github.com/dimenwarper/fanout"
    typewrite_centered(16, url, delay=0.04, color=f"{BOLD}{BRIGHT_WHITE}")

    pause(0.8)

    put(18, center_col(34), f"{GRAY}Sample. Evaluate. Evolve. Repeat.{RESET}")

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
        scene_agents()
        scene_channel()
        scene_strategies()
        scene_loop()
        scene_memory()
        scene_closing()
    finally:
        sys.stdout.write(SHOW_CURSOR + RESET + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
