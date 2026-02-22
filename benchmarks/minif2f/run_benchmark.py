#!/usr/bin/env python3
"""miniF2F benchmark runner for fanout.

Runs Lean 4 theorem proving tasks through fanout's evolutionary loop.
Requires elan/lake (Lean 4 toolchain) for evaluation.

Usage:
    uv run --extra benchmarks python benchmarks/minif2f/run_benchmark.py
    uv run --extra benchmarks python benchmarks/minif2f/run_benchmark.py --strategy rsa --rounds 5
    uv run --extra benchmarks python benchmarks/minif2f/run_benchmark.py --tasks imo_1959_p1 mathd_algebra_478
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from fanout.db.models import Run
from fanout.evaluate import evaluate_solutions
from fanout.providers.openrouter import SamplingConfig
from fanout.sample import sample as do_sample
from fanout.select import select_solutions
from fanout.store import Store
from fanout.strategies.base import get_strategy

console = Console()

BENCHMARK_DIR = Path(__file__).resolve().parent
EVAL_SCRIPT = str(BENCHMARK_DIR / "eval.py")

TASKS = {
    "imo_1959_p1": {
        "file": "tasks/imo_1959_p1.lean",
        "description": "gcd(21n+4, 14n+3) = 1",
        "difficulty": "IMO",
    },
    "imo_1963_p5": {
        "file": "tasks/imo_1963_p5.lean",
        "description": "cos(pi/7) - cos(2pi/7) + cos(3pi/7) = 1/2",
        "difficulty": "IMO",
    },
    "imo_1977_p6": {
        "file": "tasks/imo_1977_p6.lean",
        "description": "f(f(n)) < f(n+1) implies f = id",
        "difficulty": "IMO",
    },
    "imo_1990_p3": {
        "file": "tasks/imo_1990_p3.lean",
        "description": "n^2 | 2^n+1 implies n=3",
        "difficulty": "IMO",
    },
    "mathd_algebra_478": {
        "file": "tasks/mathd_algebra_478.lean",
        "description": "Basic algebra (warmup)",
        "difficulty": "MATHD",
    },
}


def build_prompt(task_name: str, task_info: dict) -> str:
    task_path = BENCHMARK_DIR / task_info["file"]
    task_source = task_path.read_text()

    return (
        f"Here is a Lean 4 theorem statement with `sorry` as a placeholder:\n\n"
        f"```lean\n{task_source}\n```\n\n"
        f"Replace `sorry` with a valid Lean 4 proof. The proof must typecheck with Mathlib.\n"
        f"Use appropriate Lean 4 tactics (omega, norm_num, ring, simp, linarith, etc.).\n\n"
        f"Output only the complete .lean file with the proof filled in. Do not include sorry."
    )


def run_task(
    task_name: str,
    task_info: dict,
    *,
    models: list[str],
    model_set: str | None = None,
    n_samples: int = 5,
    strategy: str,
    rounds: int,
    n_per_model: int,
    k: int,
    k_agg: int,
    temperature: float,
    max_tokens: int,
    verbose: bool = False,
) -> dict[str, Any]:
    console.rule(f"[bold cyan]Task: {task_name}[/]")
    console.print(f"  {task_info['description']} [{task_info['difficulty']}]")
    if model_set:
        console.print(f"  Strategy: {strategy}, Rounds: {rounds}, Model set: {model_set} (N={n_samples})")
    else:
        console.print(f"  Strategy: {strategy}, Rounds: {rounds}, Models: {models}")

    prompt = build_prompt(task_name, task_info)

    if verbose:
        console.print(f"\n  [dim]Prompt ({len(prompt)} chars):[/]")
        console.print(f"  [dim]{prompt[:200]}...[/]\n")

    store = Store()
    run = Run(prompt=prompt, total_rounds=rounds)
    store.save_run(run)

    config = SamplingConfig(
        models=models,
        temperature=temperature,
        max_tokens=max_tokens,
        n_per_model=n_per_model,
        model_set=model_set,
        n_samples=n_samples,
    )
    context: dict[str, Any] = {
        "eval_script": EVAL_SCRIPT,
        "materializer": "file",
        "file_extension": ".lean",
    }
    evaluator_names = ["script"]
    strategy_instance = get_strategy(strategy)
    current_prompt: str | list[str] = prompt
    parent_ids: list[str] | None = None

    best_score = 0.0
    round_scores: list[float] = []
    solved = False

    for rnd in range(rounds):
        console.print(f"  [dim]Round {rnd + 1}/{rounds}...[/]", end=" ")

        solutions = do_sample(current_prompt, config, store, run.id, rnd, parent_ids)

        from collections import Counter
        model_counts = Counter(s.model for s in solutions)
        models_str = ", ".join(f"{m}(x{c})" if c > 1 else m for m, c in model_counts.items())
        console.print(f"[dim]sampled \\[{models_str}][/]", end=" ")

        evals = evaluate_solutions(solutions, evaluator_names, store, context)
        selected = select_solutions(run.id, rnd, strategy, store, k=k)

        top_score = selected[0].aggregate_score if selected else 0.0
        round_scores.append(top_score)
        best_score = max(best_score, top_score)

        if top_score >= 1.0:
            console.print(f"[bold green]SOLVED (QED)[/]")
            solved = True
        else:
            console.print(f"top={top_score:.4f}")

        if verbose:
            for i, (sol, ev) in enumerate(zip(solutions, evals)):
                stderr = ev.details.get("stderr", "")
                stdout = ev.details.get("stdout", "")
                exit_code = ev.details.get("exit_code", "?")
                preview_lines = sol.output[:500].splitlines()[:15]
                preview = "\n".join(preview_lines)
                console.print(f"    [dim]Solution {i+1} [{sol.model}] score={ev.score:.4f} exit={exit_code}[/]")
                console.print(Syntax(preview, "lean4", theme="monokai", line_numbers=True, padding=(0, 2)))
                if stderr:
                    console.print(f"      [dim]stderr: {stderr}[/]")
                if ev.score == 0.0 and stdout:
                    console.print(f"      [dim]stdout: {stdout}[/]")

        store.update_run_round(run.id, rnd + 1)
        parent_ids = [s.solution.id for s in selected]

        # Stop early if solved
        if solved:
            break

        if rnd < rounds - 1:
            n_expected = config.n_samples if config.model_set else len(config.models) * config.n_per_model
            current_prompt = strategy_instance.build_prompts(
                original_prompt=prompt,
                selected=selected,
                round_num=rnd,
                n_samples=n_expected,
                k_agg=k_agg,
            )

    return {
        "task": task_name,
        "difficulty": task_info["difficulty"],
        "strategy": strategy,
        "rounds": len(round_scores),
        "best_score": best_score,
        "solved": solved,
        "round_scores": round_scores,
        "run_id": run.id,
    }


def main():
    parser = argparse.ArgumentParser(description="Run miniF2F benchmarks with fanout")
    parser.add_argument(
        "--tasks", nargs="*", default=list(TASKS.keys()),
        help=f"Tasks to run (default: all). Choices: {list(TASKS.keys())}",
    )
    parser.add_argument(
        "--strategy", "-s", nargs="*", default=["top-k", "rsa"],
        help="Strategies to compare (default: top-k rsa)",
    )
    parser.add_argument("--model", "-m", action="append", help="Model (repeatable)")
    parser.add_argument("--model-set", "-M", help="Named model set (e.g., coding, math-proving)")
    parser.add_argument("--rounds", "-r", type=int, default=5)
    parser.add_argument("-n", type=int, default=3, help="Samples per model per round")
    parser.add_argument("-N", "--n-samples", type=int, default=5, help="Total samples when using a model set (default: 5)")
    parser.add_argument("-k", type=int, default=3, help="Selection size")
    parser.add_argument("--k-agg", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--verbose", "-v", action="store_true", help="Show eval details, solution previews, and stderr")
    args = parser.parse_args()

    models = args.model or ["openai/gpt-4o-mini"]
    results: list[dict[str, Any]] = []

    for task_name in args.tasks:
        if task_name not in TASKS:
            console.print(f"[red]Unknown task: {task_name}[/]")
            continue
        for strategy in args.strategy:
            result = run_task(
                task_name,
                TASKS[task_name],
                models=models,
                model_set=args.model_set,
                n_samples=args.n_samples,
                strategy=strategy,
                rounds=args.rounds,
                n_per_model=args.n,
                k=args.k,
                k_agg=args.k_agg,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                verbose=args.verbose,
            )
            results.append(result)

    console.print()
    table = Table(title="miniF2F Results")
    table.add_column("Task", style="cyan")
    table.add_column("Difficulty")
    table.add_column("Strategy", style="magenta")
    table.add_column("Rounds", justify="right")
    table.add_column("Solved", justify="center")
    table.add_column("Per-Round Scores", style="dim")
    table.add_column("Run ID", style="dim")

    for r in results:
        scores_str = " → ".join(f"{s:.0f}" for s in r["round_scores"])
        solved_str = "[bold green]YES[/]" if r["solved"] else "[red]no[/]"
        table.add_row(
            r["task"], r["difficulty"], r["strategy"], str(r["rounds"]),
            solved_str, scores_str, r["run_id"][:8],
        )

    console.print(table)

    # Summary
    for strat in set(r["strategy"] for r in results):
        strat_results = [r for r in results if r["strategy"] == strat]
        solved = sum(1 for r in strat_results if r["solved"])
        console.print(f"  {strat}: {solved}/{len(strat_results)} solved")


if __name__ == "__main__":
    main()
