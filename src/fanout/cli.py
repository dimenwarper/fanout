"""Fanout CLI — typer entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Optional

from dotenv import load_dotenv
import typer

# Load .env from cwd (project root) before anything reads env vars
load_dotenv(Path.cwd() / ".env")
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from fanout.db.models import Run
from fanout.evaluators import list_evaluators
from fanout.materializers import list_materializers
from fanout.model_sets import load_model_sets
from fanout.providers.openrouter import SamplingConfig
from fanout.store import Store
from fanout.strategies import list_strategies

app = typer.Typer(
    name="fanout",
    help="Sample multiple LLMs, evaluate outputs, and iteratively select the best solutions.",
    no_args_is_help=True,
)
console = Console()


def _get_store() -> Store:
    return Store()


_EXT_LEXERS = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".rs": "rust", ".go": "go", ".c": "c", ".cpp": "cpp",
    ".java": "java", ".lean": "lean4", ".sh": "bash",
    ".cu": "cuda", ".html": "html", ".css": "css",
}


def _ext_to_lexer(file_ext: str) -> str:
    return _EXT_LEXERS.get(file_ext, "text")


# ── sample ────────────────────────────────────────────────

@app.command()
def sample(
    prompt: Annotated[str, typer.Argument(help="The prompt to send to models")],
    model: Annotated[Optional[list[str]], typer.Option("-m", "--model", help="Model to sample (repeatable)")] = None,
    model_set: Annotated[Optional[str], typer.Option("-M", "--model-set", help="Named model set to sample from")] = None,
    temperature: Annotated[float, typer.Option(help="Sampling temperature")] = 0.7,
    max_tokens: Annotated[int, typer.Option(help="Max tokens per response")] = 2048,
    n_samples: Annotated[int, typer.Option("-n", "--n-samples", help="Total samples per round")] = 5,
    run_id: Annotated[Optional[str], typer.Option(help="Existing run ID to add to")] = None,
    round_num: Annotated[int, typer.Option(help="Round number")] = 0,
    eval_script: Annotated[Optional[str], typer.Option("--eval-script", help="Path to eval script (implies -e script)")] = None,
    materializer: Annotated[str, typer.Option("--materializer", help="Materializer name")] = "file",
    file_ext: Annotated[str, typer.Option("--file-ext", help="File extension for file materializer")] = ".py",
    eval_concurrency: Annotated[int, typer.Option("-p", "--eval-concurrency", help="Max parallel evaluations")] = 1,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Show solution previews with syntax highlighting")] = False,
    api_key: Annotated[Optional[str], typer.Option(envvar="OPENROUTER_API_KEY", help="OpenRouter API key")] = None,
) -> None:
    """Fan out a prompt to one or more models."""
    from collections import Counter

    from fanout.evaluate import evaluate_solutions
    from fanout.sample import sample as do_sample

    store = _get_store()
    models = model or ["openai/gpt-4o-mini"]

    if run_id is None:
        run = Run(prompt=prompt)
        store.save_run(run)
        run_id = run.id
        console.print(f"[bold green]Created run:[/] {run_id}")

    config = SamplingConfig(
        models=models, temperature=temperature,
        max_tokens=max_tokens,
        model_set=model_set, n_samples=n_samples,
    )
    solutions = do_sample(prompt, config, store, run_id, round_num, api_key=api_key)

    # Show which models were sampled
    model_counts = Counter(s.model for s in solutions)
    models_str = ", ".join(f"{m}(x{c})" if c > 1 else m for m, c in model_counts.items())
    console.print(f"Sampled {len(solutions)} solutions [dim]\\[{models_str}][/]")

    # If --eval-script is provided, run the script evaluator automatically
    evals = None
    if eval_script:
        context = {
            "eval_script": eval_script,
            "materializer": materializer,
            "file_extension": file_ext,
        }
        evals = evaluate_solutions(solutions, ["script"], store, context, concurrency=eval_concurrency)
        console.print(f"[bold green]Script evaluator:[/] {len(evals)} evaluations")

    if verbose:
        lexer = _ext_to_lexer(file_ext)
        for i, sol in enumerate(solutions):
            score_str = ""
            if evals:
                score_str = f" score={evals[i].score:.4f}"
            console.print(f"  [dim]Solution {i+1} [{sol.model}]{score_str} latency={sol.latency_ms:.0f}ms[/]")
            preview_lines = sol.output[:500].splitlines()[:15]
            preview = "\n".join(preview_lines)
            console.print(Syntax(preview, lexer, theme="monokai", line_numbers=True, padding=(0, 2)))
            if evals:
                stderr = evals[i].details.get("stderr", "")
                if stderr:
                    console.print(f"      [dim]stderr: {stderr}[/]")
    else:
        table = Table(title=f"Sampled {len(solutions)} solutions")
        table.add_column("ID", style="cyan")
        table.add_column("Model")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Output (preview)")
        for sol in solutions:
            table.add_row(
                sol.id, sol.model, f"{sol.latency_ms:.0f}",
                str(sol.prompt_tokens + sol.completion_tokens),
                sol.output[:80] + ("..." if len(sol.output) > 80 else ""),
            )
        console.print(table)


# ── evaluate ──────────────────────────────────────────────

@app.command()
def evaluate(
    run_id: Annotated[str, typer.Argument(help="Run ID to evaluate")],
    evaluator: Annotated[Optional[list[str]], typer.Option("-e", "--evaluator", help="Evaluator to use (repeatable)")] = None,
    round_num: Annotated[Optional[int], typer.Option(help="Round to evaluate")] = None,
    reference: Annotated[Optional[str], typer.Option(help="Reference answer for accuracy evaluator")] = None,
) -> None:
    """Evaluate solutions in a run."""
    from fanout.evaluate import evaluate_solutions

    store = _get_store()
    evaluator_names = evaluator or ["latency", "cost"]
    solutions = store.get_solutions_for_run(run_id, round_num)

    if not solutions:
        console.print("[yellow]No solutions found for this run/round.[/]")
        raise typer.Exit(1)

    context = {}
    if reference:
        context["reference"] = reference

    evals = evaluate_solutions(solutions, evaluator_names, store, context or None)
    console.print(f"[bold green]Created {len(evals)} evaluations[/] across {len(solutions)} solutions")

    # Show scores
    scored = store.get_solutions_with_scores(run_id, round_num)
    table = Table(title="Scores")
    table.add_column("Solution ID", style="cyan")
    table.add_column("Model")
    table.add_column("Aggregate", justify="right", style="bold")
    ev_names = sorted({e.evaluator for e in evals})
    for name in ev_names:
        table.add_column(name, justify="right")
    for s in scored:
        row = [s.solution.id, s.solution.model, f"{s.aggregate_score:.3f}"]
        for name in ev_names:
            row.append(f"{s.scores_by_evaluator.get(name, 0):.3f}")
        table.add_row(*row)
    console.print(table)


# ── select ────────────────────────────────────────────────

@app.command()
def select(
    run_id: Annotated[str, typer.Argument(help="Run ID to select from")],
    strategy: Annotated[str, typer.Option("-s", "--strategy", help="Selection strategy")] = "top-k",
    round_num: Annotated[int, typer.Option(help="Round to select from")] = 0,
    k: Annotated[int, typer.Option(help="Number to select (for top-k/weighted)")] = 3,
) -> None:
    """Select best solutions from a round."""
    from fanout.select import select_solutions

    store = _get_store()
    selected = select_solutions(run_id, round_num, strategy, store, k=k)

    if not selected:
        console.print("[yellow]No scored solutions found.[/]")
        raise typer.Exit(1)

    table = Table(title=f"Selected {len(selected)} solutions ({strategy})")
    table.add_column("Solution ID", style="cyan")
    table.add_column("Model")
    table.add_column("Score", justify="right", style="bold")
    table.add_column("Output (preview)")
    for s in selected:
        table.add_row(
            s.solution.id, s.solution.model, f"{s.aggregate_score:.3f}",
            s.solution.output[:80] + ("..." if len(s.solution.output) > 80 else ""),
        )
    console.print(table)


# ── run (full loop) ──────────────────────────────────────

@app.command(name="run")
def run_loop(
    prompt: Annotated[str, typer.Argument(help="The prompt to send")],
    model: Annotated[Optional[list[str]], typer.Option("-m", "--model", help="Model (repeatable)")] = None,
    model_set: Annotated[Optional[str], typer.Option("-M", "--model-set", help="Named model set to sample from")] = None,
    evaluator: Annotated[Optional[list[str]], typer.Option("-e", "--evaluator", help="Evaluator (repeatable)")] = None,
    strategy: Annotated[str, typer.Option("-s", "--strategy", help="Selection strategy")] = "top-k",
    rounds: Annotated[int, typer.Option("-r", "--rounds", help="Number of evolutionary rounds")] = 1,
    n_samples: Annotated[int, typer.Option("-n", "--n-samples", help="Total samples per round")] = 5,
    k: Annotated[int, typer.Option(help="Selection size")] = 3,
    temperature: Annotated[float, typer.Option(help="Sampling temperature")] = 0.7,
    max_tokens: Annotated[int, typer.Option(help="Max tokens per response")] = 2048,
    reference: Annotated[Optional[str], typer.Option(help="Reference answer for accuracy evaluator")] = None,
    eval_script: Annotated[Optional[str], typer.Option("--eval-script", help="Path to eval script (implies -e script)")] = None,
    materializer: Annotated[str, typer.Option("--materializer", help="Materializer name")] = "file",
    file_ext: Annotated[str, typer.Option("--file-ext", help="File extension for file materializer")] = ".py",
    k_agg: Annotated[int, typer.Option("--k-agg", help="Number of parent solutions per aggregation prompt (RSA)")] = 3,
    eval_concurrency: Annotated[int, typer.Option("-p", "--eval-concurrency", help="Max parallel evaluations")] = 1,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Show per-solution details with syntax-highlighted previews")] = False,
    api_key: Annotated[Optional[str], typer.Option(envvar="OPENROUTER_API_KEY")] = None,
) -> None:
    """Full evolutionary loop: sample → evaluate → select × N rounds."""
    from collections import Counter

    from fanout.evaluate import evaluate_solutions
    from fanout.sample import sample as do_sample
    from fanout.select import select_solutions
    from fanout.strategies.base import get_strategy

    store = _get_store()
    models = model or ["openai/gpt-4o-mini"]
    evaluator_names = evaluator or ["latency", "cost"]

    # If --eval-script provided, add script evaluator
    if eval_script:
        if "script" not in evaluator_names:
            evaluator_names.append("script")

    run = Run(prompt=prompt, total_rounds=rounds)
    store.save_run(run)
    if model_set:
        console.print(f"[bold green]Run {run.id}[/] — {rounds} round(s), model set: {model_set} (n={n_samples})")
    else:
        console.print(f"[bold green]Run {run.id}[/] — {rounds} round(s), {len(models)} model(s), n={n_samples}")

    config = SamplingConfig(
        models=models, temperature=temperature,
        max_tokens=max_tokens,
        model_set=model_set, n_samples=n_samples,
    )
    context: dict[str, Any] | None = {}
    if reference:
        context["reference"] = reference
    if eval_script:
        context["eval_script"] = eval_script
        context["materializer"] = materializer
        context["file_extension"] = file_ext
    context = context or None
    parent_ids: list[str] | None = None
    strategy_instance = get_strategy(strategy)
    current_prompt: str | list[str] = prompt
    best_score = 0.0
    lexer = _ext_to_lexer(file_ext)

    for rnd in range(rounds):
        console.rule(f"[bold]Round {rnd + 1}/{rounds}[/]")

        # Sample
        solutions = do_sample(current_prompt, config, store, run.id, rnd, parent_ids, api_key)

        model_counts = Counter(s.model for s in solutions)
        models_str = ", ".join(f"{m}(x{c})" if c > 1 else m for m, c in model_counts.items())
        console.print(f"  Sampled {len(solutions)} solutions [dim]\\[{models_str}][/]")

        # Evaluate
        evals = evaluate_solutions(solutions, evaluator_names, store, context, concurrency=eval_concurrency)
        console.print(f"  Ran {len(evals)} evaluations")

        # Select
        selected = select_solutions(run.id, rnd, strategy, store, k=k)
        top_score = selected[0].aggregate_score if selected else 0.0
        best_score = max(best_score, top_score)
        console.print(f"  Selected {len(selected)} ({strategy}) top={top_score:.4f} best={best_score:.4f}")

        # Show top
        for i, s in enumerate(selected[:3], 1):
            console.print(f"  #{i} [{s.solution.model}] score={s.aggregate_score:.3f}")

        if verbose:
            for i, sol in enumerate(solutions):
                ev = evals[i] if i < len(evals) else None
                score_str = f" score={ev.score:.4f}" if ev else ""
                exit_str = f" exit={ev.details.get('exit_code', '?')}" if ev else ""
                console.print(f"    [dim]Solution {i+1} [{sol.model}]{score_str}{exit_str}[/]")
                preview_lines = sol.output[:500].splitlines()[:15]
                preview = "\n".join(preview_lines)
                console.print(Syntax(preview, lexer, theme="monokai", line_numbers=True, padding=(0, 2)))
                if ev:
                    stderr = ev.details.get("stderr", "")
                    stdout = ev.details.get("stdout", "")
                    if stderr:
                        console.print(f"      [dim]stderr: {stderr}[/]")
                    if ev.score == 0.0 and stdout:
                        console.print(f"      [dim]stdout: {stdout}[/]")

        store.update_run_round(run.id, rnd + 1)
        parent_ids = [s.solution.id for s in selected]

        # Build prompts for next round
        if rnd < rounds - 1:
            current_prompt = strategy_instance.build_prompts(
                original_prompt=prompt,
                selected=selected,
                round_num=rnd,
                n_samples=config.n_samples,
                k_agg=k_agg,
            )

    console.print(f"\n[bold green]Done.[/] Run ID: {run.id} best={best_score:.4f}")


# ── store (inspect) ──────────────────────────────────────

@app.command(name="store")
def store_inspect(
    run_id: Annotated[Optional[str], typer.Argument(help="Run ID to inspect")] = None,
) -> None:
    """List runs or inspect a specific run."""
    store = _get_store()

    if run_id is None:
        runs = store.list_runs()
        if not runs:
            console.print("[yellow]No runs found.[/]")
            return
        table = Table(title=f"{len(runs)} runs")
        table.add_column("ID", style="cyan")
        table.add_column("Prompt")
        table.add_column("Rounds", justify="right")
        table.add_column("Created")
        for r in runs:
            table.add_row(r.id, r.prompt[:60], f"{r.current_round}/{r.total_rounds}", str(r.created_at))
        console.print(table)
    else:
        scored = store.get_solutions_with_scores(run_id)
        if not scored:
            console.print("[yellow]No solutions found.[/]")
            return
        table = Table(title=f"Run {run_id} — {len(scored)} solutions")
        table.add_column("ID", style="cyan")
        table.add_column("Round", justify="right")
        table.add_column("Model")
        table.add_column("Score", justify="right", style="bold")
        table.add_column("Output (preview)")
        for s in scored:
            table.add_row(
                s.solution.id, str(s.solution.round_num), s.solution.model,
                f"{s.aggregate_score:.3f}",
                s.solution.output[:60] + ("..." if len(s.solution.output) > 60 else ""),
            )
        console.print(table)


# ── list-evaluators ──────────────────────────────────────

@app.command(name="list-evaluators")
def list_evaluators_cmd() -> None:
    """List available evaluators."""
    table = Table(title="Evaluators")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    for name, cls in sorted(list_evaluators().items()):
        table.add_row(name, cls.description)
    console.print(table)


# ── list-materializers ────────────────────────────────────

@app.command(name="list-materializers")
def list_materializers_cmd() -> None:
    """List available materializers."""
    table = Table(title="Materializers")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    for name, cls in sorted(list_materializers().items()):
        table.add_row(name, cls.description)
    console.print(table)


# ── list-strategies ──────────────────────────────────────

@app.command(name="list-strategies")
def list_strategies_cmd() -> None:
    """List available selection strategies."""
    table = Table(title="Strategies")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    for name, cls in sorted(list_strategies().items()):
        table.add_row(name, cls.description)
    console.print(table)


# ── list-model-sets ──────────────────────────────────────

@app.command(name="list-model-sets")
def list_model_sets_cmd() -> None:
    """List available model sets (builtins + user-defined)."""
    sets = load_model_sets()

    for name, ms in sorted(sets.items()):
        table = Table(title=f"Model set: {name}")
        table.add_column("Model", style="cyan")
        table.add_column("Weight", justify="right")
        total_weight = sum(e.weight for e in ms.models)
        for entry in ms.models:
            pct = entry.weight / total_weight * 100
            table.add_row(entry.model, f"{entry.weight:.1f} ({pct:.0f}%)")
        console.print(table)
        console.print()
