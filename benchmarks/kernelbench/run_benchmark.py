#!/usr/bin/env python3
"""KernelBench benchmark runner for fanout.

Runs CUDA kernel optimization tasks through fanout's evolutionary loop.
Requires a CUDA-capable GPU with PyTorch installed.

Usage:
    uv run --extra benchmarks python benchmarks/kernelbench/run_benchmark.py
    uv run --extra benchmarks python benchmarks/kernelbench/run_benchmark.py --strategy rsa --rounds 3
    uv run --extra benchmarks python benchmarks/kernelbench/run_benchmark.py --tasks matmul relu
"""

from __future__ import annotations

import argparse
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from rich.console import Console
from rich.table import Table

from fanout.workflow import Workflow, sample_step, evaluate_step, select_step, evolve_step, launch_step

console = Console()

BENCHMARK_DIR = Path(__file__).resolve().parent
EVAL_SCRIPT = BENCHMARK_DIR / "eval.py"

TASKS = {
    "matmul": {
        "file": "tasks/matmul.py",
        "description": "Square matrix multiplication (4096x4096)",
    },
    "relu": {
        "file": "tasks/relu.py",
        "description": "Elementwise ReLU on large tensor",
    },
    "layernorm": {
        "file": "tasks/layernorm.py",
        "description": "Layer normalization over 3D feature shape",
    },
    "conv2d": {
        "file": "tasks/conv2d.py",
        "description": "AlexNet-style 2D convolution",
    },
    "sum_reduce": {
        "file": "tasks/sum_reduce.py",
        "description": "Sum reduction over one dimension",
    },
}


def build_prompt(task_name: str, task_info: dict) -> str:
    task_path = BENCHMARK_DIR / task_info["file"]
    task_source = task_path.read_text()

    return (
        f"Here is a PyTorch Model class:\n\n"
        f"```python\n{task_source}\n```\n\n"
        f"Write an optimized `ModelNew` class that produces identical outputs to `Model` but runs faster "
        f"using custom CUDA kernels. You can use:\n"
        f"- torch.autograd.Function with inline CUDA via torch.utils.cpp_extension\n"
        f"- Triton kernels (@triton.jit)\n"
        f"- Any PyTorch-compatible approach\n\n"
        f"The ModelNew class must have the same __init__ signature and forward() interface as Model."
    )


def make_task_eval_script(task_name: str) -> Path:
    """Create a wrapper script that passes the task file to eval.py."""
    task_path = BENCHMARK_DIR / TASKS[task_name]["file"]
    wrapper = tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", prefix=f"eval_{task_name}_", delete=False,
    )
    wrapper.write(f"#!/bin/bash\n")
    wrapper.write(f'exec python3 "{EVAL_SCRIPT}" "$1" "{task_path}"\n')
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
    verbose: bool = False,
    full: bool = False,
    mode: str = "sample",
    n_agents: int = 3,
    max_steps: int = 10,
) -> dict[str, Any]:
    console.rule(f"[bold cyan]Task: {task_name}[/]")
    console.print(f"  {task_info['description']}")
    if mode == "agent":
        console.print(f"  Mode: agent, Agents: {n_agents}, Max steps: {max_steps}")
    elif model_set:
        console.print(f"  Strategy: {strategy}, Rounds: {rounds}, Model set: {model_set} (N={n_samples})")
    else:
        console.print(f"  Strategy: {strategy}, Rounds: {rounds}, Models: {models}")

    prompt = build_prompt(task_name, task_info)
    eval_wrapper = make_task_eval_script(task_name)

    try:
        if mode == "agent":
            wf = Workflow(steps=[launch_step, select_step])
            effective_rounds = 1
        else:
            wf = Workflow(steps=[sample_step, evaluate_step, select_step, evolve_step])
            effective_rounds = rounds
        result = wf.run(
            prompt=prompt,
            models=models,
            model_set=model_set,
            n_samples=n_samples,
            rounds=effective_rounds,
            strategy=strategy,
            k=k,
            k_agg=k_agg,
            temperature=temperature,
            max_tokens=max_tokens,
            solution_format=solution_format,
            eval_script=str(eval_wrapper),
            eval_context={"file_extension": ".py"},
            eval_concurrency=eval_concurrency,
            verbose=verbose,
            full=full,
            console=console,
            n_agents=n_agents,
            max_steps=max_steps,
        )

        return {
            "task": task_name,
            "strategy": strategy,
            "rounds": rounds,
            "best_score": result.best_score,
            "round_scores": result.round_scores,
            "run_id": result.run_id,
        }
    finally:
        eval_wrapper.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Run KernelBench benchmarks with fanout")
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
    parser.add_argument("--rounds", "-r", type=int, default=3)
    parser.add_argument("-n", "--n-samples", type=int, default=5, help="Total samples per round (default: 5)")
    parser.add_argument("-k", type=int, default=3, help="Selection size")
    parser.add_argument("--k-agg", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--solution-format", default="code", help="Solution format: code, diff, raw (default: code)")
    parser.add_argument("-p", "--eval-concurrency", type=int, default=1, help="Max parallel evaluations (default: 1)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show eval details, solution previews, and stderr")
    parser.add_argument("--full", action="store_true", help="Show full solutions (not truncated)")
    parser.add_argument("--mode", choices=["sample", "agent"], default="sample", help="Workflow mode (default: sample)")
    parser.add_argument("--n-agents", type=int, default=3, help="Number of agents for agent mode (default: 3)")
    parser.add_argument("--max-steps", type=int, default=10, help="Max steps per agent (default: 10)")
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
                k=args.k,
                k_agg=args.k_agg,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                solution_format=args.solution_format,
                eval_concurrency=args.eval_concurrency,
                verbose=args.verbose,
                full=args.full,
                mode=args.mode,
                n_agents=args.n_agents,
                max_steps=args.max_steps,
            )
            results.append(result)

    console.print()
    table = Table(title="KernelBench Results")
    table.add_column("Task", style="cyan")
    table.add_column("Strategy", style="magenta")
    table.add_column("Rounds", justify="right")
    table.add_column("Best Score", justify="right", style="bold green")
    table.add_column("Per-Round Scores", style="dim")
    table.add_column("Run ID", style="dim")

    for r in results:
        scores_str = " → ".join(f"{s:.3f}" for s in r["round_scores"])
        table.add_row(
            r["task"], r["strategy"], str(r["rounds"]),
            f"{r['best_score']:.4f}", scores_str, r["run_id"][:8],
        )
    console.print(table)


if __name__ == "__main__":
    main()
