#!/usr/bin/env python3
"""PDE Solvers benchmark runner for fanout.

Runs each PDE task through fanout's evolutionary loop and reports scores.
Demonstrates that multi-model evolutionary search outperforms single-shot
LLM generation on physics simulation problems.

Usage:
    uv run --extra benchmarks python benchmarks/pde-solvers/run_benchmark.py
    uv run --extra benchmarks python benchmarks/pde-solvers/run_benchmark.py --tasks burgers_1d -r 3
    uv run --extra benchmarks python benchmarks/pde-solvers/run_benchmark.py --strategy rsa --rounds 5
"""

from __future__ import annotations

import argparse
import asyncio
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from rich.console import Console
from rich.table import Table

from fanout.report import generate_summary, print_memory_summary, save_record
from fanout.store import Store
from fanout.workflow import SampleWorkflow, LaunchWorkflow

console = Console()

BENCHMARK_DIR = Path(__file__).resolve().parent
EVAL_SCRIPT = BENCHMARK_DIR / "eval.py"

TASKS = {
    "burgers_1d": {
        "file": "tasks/burgers_1d.py",
        "function": "solve_pde",
        "description": "1D Burgers equation (ν=0.01, periodic, 128 pts) — Tier 1: Easy",
        "timeout": 60,
        "benchmark": 0.84,
    },
    "navier_stokes_2d": {
        "file": "tasks/navier_stokes_2d.py",
        "function": "solve_pde",
        "description": "2D Navier-Stokes vorticity (ν=1e-3, 64×64, periodic) — Tier 2: Medium",
        "timeout": 300,
        "benchmark": 0.90,
    },
    "ks_1d": {
        "file": "tasks/ks_1d.py",
        "function": "solve_pde",
        "description": "1D Kuramoto-Sivashinsky (chaotic, 256 pts) — Tier 3: Hard",
        "timeout": 300,
        "benchmark": 0.60,
    },
}


def build_prompt(task_name: str, task_info: dict) -> str:
    """Build the LLM prompt from a task file."""
    task_path = BENCHMARK_DIR / task_info["file"]
    task_source = task_path.read_text()

    return (
        f"{task_source}\n\n"
        f"Improve the function `{task_info['function']}()` to achieve the best possible score. "
        f"The score is computed as 1/(1 + avg_nRMSE) where nRMSE is normalized RMSE against "
        f"a high-resolution spectral reference solution.\n\n"
        f"The baseline score is approximately {task_info['benchmark']}. "
        f"Better numerical methods (spectral, implicit, higher-order) will score higher.\n\n"
        f"Keep the same function name and signature. You may use numpy and scipy."
    )


def make_task_eval_script(task_name: str) -> Path:
    """Create a wrapper script that passes the task name to eval.py."""
    wrapper = tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", prefix=f"eval_{task_name}_", delete=False,
    )
    wrapper.write(f"#!/bin/bash\n")
    wrapper.write(f'exec python3 "{EVAL_SCRIPT}" "$1" {task_name}\n')
    wrapper.close()
    os.chmod(wrapper.name, os.stat(wrapper.name).st_mode | stat.S_IEXEC)
    return Path(wrapper.name)


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
    eval_timeout: int | None = None,
    verbose: bool = False,
    full: bool = False,
    mode: str = "sample",
    n_agents: int = 3,
    max_steps: int = 10,
    use_memory: bool = False,
    store: Store | None = None,
) -> dict[str, Any]:
    """Run a single benchmark task through fanout and return results."""
    if eval_timeout is None:
        eval_timeout = task_info.get("timeout", 60)

    console.rule(f"[bold cyan]Task: {task_name}[/]")
    console.print(f"  {task_info['description']}")
    mem_str = ", [bold]Memory:[/] enabled" if use_memory else ""
    if mode == "agent":
        console.print(f"  [bold]Mode:[/] agent, [bold]Strategy:[/] {strategy}, [bold]Agents:[/] {n_agents}, [bold]Max steps:[/] {max_steps}{mem_str}")
    elif model_set:
        console.print(f"  [bold]Strategy:[/] {strategy}, [bold]Rounds:[/] {rounds}, [bold]Model set:[/] {model_set} (N={n_samples}){mem_str}")
    else:
        console.print(f"  [bold]Strategy:[/] {strategy}, [bold]Rounds:[/] {rounds}, [bold]Models:[/] {models}{mem_str}")

    prompt = build_prompt(task_name, task_info)
    eval_wrapper = make_task_eval_script(task_name)

    try:
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
                eval_script=str(eval_wrapper),
                eval_context={"file_extension": ".py", "eval_timeout": eval_timeout},
                verbose=verbose,
                full=full,
                console=console,
                use_memory=use_memory,
                store=store,
            )
        else:
            wf = SampleWorkflow()
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
                eval_script=str(eval_wrapper),
                eval_context={"file_extension": ".py", "eval_timeout": eval_timeout},
                eval_concurrency=eval_concurrency,
                verbose=verbose,
                full=full,
                console=console,
                use_memory=use_memory,
                store=store,
            )

        return {
            "task": task_name,
            "strategy": strategy,
            "mode": mode,
            "iterations": max_steps if mode == "agent" else rounds,
            "best_score": result.best_score,
            "benchmark": task_info.get("benchmark"),
            "round_scores": result.round_scores,
            "run_id": result.run_id,
        }
    finally:
        eval_wrapper.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Run PDE Solvers benchmarks with fanout")
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
    parser.add_argument("--rounds", "-r", type=int, default=3, help="Rounds per run (default: 3)")
    parser.add_argument("-n", "--n-samples", type=int, default=5, help="Total samples per round (default: 5)")
    parser.add_argument("-k", type=int, default=3, help="Selection size (default: 3)")
    parser.add_argument("--k-agg", type=int, default=6, help="RSA aggregation size (default: 6)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--solution-format", default="code", help="Solution format: code, diff, raw (default: code)")
    parser.add_argument("-p", "--eval-concurrency", type=int, default=1, help="Max parallel evaluations (default: 1)")
    parser.add_argument("--eval-timeout", type=int, default=None, help="Override timeout per evaluation in seconds")
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

    # Summary table
    console.print()
    table = Table(title="PDE Solvers Benchmark Results")
    table.add_column("Task", style="cyan")
    table.add_column("Strategy", style="magenta")
    table.add_column("Mode")
    table.add_column("Rounds/Steps", justify="right")
    table.add_column("Best Score", justify="right", style="bold green")
    table.add_column("Benchmark", justify="right", style="dim")
    table.add_column("Per-Round Scores", style="dim")
    table.add_column("Run ID", style="dim")

    for r in results:
        scores_str = " → ".join(f"{s:.3f}" for s in r["round_scores"])
        bench = str(r.get("benchmark", "—"))
        table.add_row(
            r["task"], r["strategy"], r["mode"], str(r["iterations"]),
            f"{r['best_score']:.4f}", bench, scores_str, r["run_id"][:8],
        )
    console.print(table)

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
