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
import asyncio
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from rich.console import Console
from rich.table import Table

from fanout.report import generate_summary, print_memory_summary, save_record
from fanout.store import Store
from fanout.workflow import SampleWorkflow, LaunchWorkflow, WorkflowContext

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
        f"Do not include sorry."
    )


def stop_if_solved(ctx: WorkflowContext) -> None:
    """Stop early when proof is found."""
    if ctx.best_score >= 1.0:
        ctx.stop = True


def run_task(
    task_name: str,
    task_info: dict,
    *,
    models: list[str],
    model_set: str | None = None,
    n_samples: int = 5,
    strategy: str,
    rounds: int,
    k: int,
    k_agg: int,
    temperature: float,
    max_tokens: int,
    solution_format: str = "code",
    eval_concurrency: int = 1,
    eval_timeout: int = 120,
    verbose: bool = False,
    full: bool = False,
    mode: str = "sample",
    n_agents: int = 3,
    max_steps: int = 10,
    use_memory: bool = False,
    store: Store | None = None,
) -> dict[str, Any]:
    console.rule(f"[bold cyan]Task: {task_name}[/]")
    console.print(f"  {task_info['description']} [{task_info['difficulty']}]")
    mem_str = ", [bold]Memory:[/] enabled" if use_memory else ""
    if mode == "agent":
        console.print(f"  [bold]Mode:[/] agent, [bold]Strategy:[/] {strategy}, [bold]Agents:[/] {n_agents}, [bold]Max steps:[/] {max_steps}{mem_str}")
    elif model_set:
        console.print(f"  [bold]Strategy:[/] {strategy}, [bold]Rounds:[/] {rounds}, [bold]Model set:[/] {model_set} (N={n_samples}){mem_str}")
    else:
        console.print(f"  [bold]Strategy:[/] {strategy}, [bold]Rounds:[/] {rounds}, [bold]Models:[/] {models}{mem_str}")

    prompt = build_prompt(task_name, task_info)

    if mode == "agent":
        wf = LaunchWorkflow()
        result = wf.run(
            prompt=prompt,
            models=models,
            model_set=model_set,
            n_samples=n_samples,
            n_agents=n_agents,
            max_steps=max_steps,
            strategy=strategy,
            k=k,
            temperature=temperature,
            max_tokens=max_tokens,
            solution_format=solution_format,
            eval_script=EVAL_SCRIPT,
            eval_context={"file_extension": ".lean", "eval_timeout": eval_timeout},
            verbose=verbose,
            full=full,
            console=console,
            syntax_lang="lean4",
            use_memory=use_memory,
            store=store,
        )
    else:
        wf = SampleWorkflow(extra_steps=[stop_if_solved])
        result = wf.run(
            prompt=prompt,
            models=models,
            model_set=model_set,
            n_samples=n_samples,
            rounds=rounds,
            strategy=strategy,
            k=k,
            k_agg=k_agg,
            temperature=temperature,
            max_tokens=max_tokens,
            solution_format=solution_format,
            eval_script=EVAL_SCRIPT,
            eval_context={"file_extension": ".lean", "eval_timeout": eval_timeout},
            eval_concurrency=eval_concurrency,
            verbose=verbose,
            full=full,
            console=console,
            syntax_lang="lean4",
            use_memory=use_memory,
            store=store,
        )

    solved = result.best_score >= 1.0

    return {
        "task": task_name,
        "difficulty": task_info["difficulty"],
        "strategy": strategy,
        "mode": mode,
        "iterations": max_steps if mode == "agent" else len(result.round_scores),
        "best_score": result.best_score,
        "solved": solved,
        "round_scores": result.round_scores,
        "run_id": result.run_id,
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
    parser.add_argument("-n", "--n-samples", type=int, default=5, help="Total samples per round (default: 5)")
    parser.add_argument("-k", type=int, default=3, help="Selection size")
    parser.add_argument("--k-agg", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--solution-format", default="code", help="Solution format: code, diff, raw (default: code)")
    parser.add_argument("-p", "--eval-concurrency", type=int, default=1, help="Max parallel evaluations (default: 1)")
    parser.add_argument("--eval-timeout", type=int, default=120, help="Timeout per evaluation in seconds (default: 120)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show eval details, solution previews, and stderr")
    parser.add_argument("--full", action="store_true", help="Show full solutions (not truncated)")
    parser.add_argument("--mode", choices=["sample", "agent"], default="sample", help="Workflow mode (default: sample)")
    parser.add_argument("--n-agents", type=int, default=3, help="Number of agents for agent mode (default: 3)")
    parser.add_argument("--max-steps", type=int, default=10, help="Max steps per agent (default: 10)")
    parser.add_argument("--memory", action="store_true", default=False, help="Enable shared memory bank for agents (default: off)")
    parser.add_argument("--record", nargs="?", const=True, default=None, metavar="NAME", help="Save to runs/ directory (optional name, e.g. --record haiku-no-memory)")
    parser.add_argument("--summary-model", default="anthropic/claude-sonnet-4-5", help="Model for LLM summary (default: anthropic/claude-sonnet-4-5)")
    args = parser.parse_args()

    models = args.model or ["openai/gpt-4o-mini"]
    results: list[dict[str, Any]] = []
    shared_store = Store()

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
                k=args.k,
                k_agg=args.k_agg,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                solution_format=args.solution_format,
                eval_concurrency=args.eval_concurrency,
                eval_timeout=args.eval_timeout,
                verbose=args.verbose,
                full=args.full,
                mode=args.mode,
                n_agents=args.n_agents,
                max_steps=args.max_steps,
                use_memory=args.memory,
                store=shared_store,
            )
            results.append(result)

    console.print()
    table = Table(title="miniF2F Results")
    table.add_column("Task", style="cyan")
    table.add_column("Difficulty")
    table.add_column("Strategy", style="magenta")
    table.add_column("Mode")
    table.add_column("Rounds/Steps", justify="right")
    table.add_column("Solved", justify="center")
    table.add_column("Per-Round Scores", style="dim")
    table.add_column("Run ID", style="dim")

    for r in results:
        scores_str = " → ".join(f"{s:.0f}" for s in r["round_scores"])
        solved_str = "[bold green]YES[/]" if r["solved"] else "[red]no[/]"
        table.add_row(
            r["task"], r["difficulty"], r["strategy"], r["mode"], str(r["iterations"]),
            solved_str, scores_str, r["run_id"][:8],
        )

    console.print(table)

    # Summary
    for strat in set(r["strategy"] for r in results):
        strat_results = [r for r in results if r["strategy"] == strat]
        solved = sum(1 for r in strat_results if r["solved"])
        console.print(f"  {strat}: {solved}/{len(strat_results)} solved")

    if args.record is not None and results:
        run_name = args.record if isinstance(args.record, str) else None
        summary = asyncio.run(
            generate_summary(results, shared_store, model=args.summary_model)
        )
        console.print(f"\n[bold]Summary:[/]\n{summary}")
        path = save_record(results, shared_store, BENCHMARK_DIR / "runs", name=run_name, summary=summary)
        console.print(f"\n[bold]Recorded to:[/] {path}")


if __name__ == "__main__":
    main()
