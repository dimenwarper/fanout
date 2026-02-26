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
from fanout.solution_format import extract_solution
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
    max_tokens: Annotated[int, typer.Option(help="Max tokens per response")] = 16384,
    n_samples: Annotated[int, typer.Option("-n", "--n-samples", help="Total samples per round")] = 5,
    run_id: Annotated[Optional[str], typer.Option(help="Existing run ID to add to")] = None,
    round_num: Annotated[int, typer.Option(help="Round number")] = 0,
    eval_script: Annotated[Optional[str], typer.Option("--eval-script", help="Path to eval script (implies -e script)")] = None,
    materializer: Annotated[str, typer.Option("--materializer", help="Materializer name")] = "file",
    file_ext: Annotated[str, typer.Option("--file-ext", help="File extension for file materializer")] = ".py",
    solution_format: Annotated[str, typer.Option("--solution-format", help="Solution format: code, diff, raw")] = "code",
    eval_concurrency: Annotated[int, typer.Option("-p", "--eval-concurrency", help="Max parallel evaluations")] = 1,
    eval_timeout: Annotated[int, typer.Option("--eval-timeout", help="Timeout per evaluation in seconds")] = 60,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Show solution previews with syntax highlighting")] = False,
    full: Annotated[bool, typer.Option("--full", help="Show full solutions (not truncated)")] = False,
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
        solution_format=solution_format,
    )
    if full:
        from fanout.solution_format import get_format
        fmt = get_format(solution_format)
        if fmt.system_prompt:
            console.print(f"\n[bold]System prompt:[/]")
            console.print(Syntax(fmt.system_prompt, "text", theme="monokai", padding=(0, 2)))
        console.print(f"\n[bold]Prompt ({len(prompt)} chars):[/]")
        console.print(Syntax(prompt, "text", theme="monokai", padding=(0, 2)))
        if fmt.prompt_suffix:
            console.print(f"\n[bold]Prompt suffix (appended to all user prompts):[/]")
            console.print(Syntax(fmt.prompt_suffix, "text", theme="monokai", padding=(0, 2)))
        console.print()

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
            "eval_timeout": eval_timeout,
        }
        evals = evaluate_solutions(solutions, ["script"], store, context, concurrency=eval_concurrency)
        console.print(f"[bold green]Script evaluator:[/] {len(evals)} evaluations")

    if full:
        lexer = _ext_to_lexer(file_ext)
        for i, sol in enumerate(solutions):
            score_str = ""
            if evals:
                score_str = f" score={evals[i].score:.4f}"
            console.print(f"  [dim]Solution {i+1} [{sol.model}]{score_str} latency={sol.latency_ms:.0f}ms[/]")
            preview = sol.output
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
            extracted = extract_solution(sol.output)
            table.add_row(
                sol.id, sol.model, f"{sol.latency_ms:.0f}",
                str(sol.prompt_tokens + sol.completion_tokens),
                extracted[:80] + ("..." if len(extracted) > 80 else ""),
            )
        console.print(table)


# ── launch ────────────────────────────────────────────────

@app.command()
def launch(
    prompt: Annotated[str, typer.Argument(help="The prompt to send to agents")],
    model: Annotated[Optional[list[str]], typer.Option("-m", "--model", help="Model to use (repeatable)")] = None,
    model_set: Annotated[Optional[str], typer.Option("-M", "--model-set", help="Named model set")] = None,
    n_agents: Annotated[int, typer.Option("-n", "--n-agents", help="Number of agents to launch")] = 3,
    max_steps: Annotated[int, typer.Option("--max-steps", help="Max agent iterations")] = 10,
    eval_script: Annotated[Optional[str], typer.Option("--eval-script", help="Path to eval script")] = None,
    materializer: Annotated[str, typer.Option("--materializer", help="Materializer name")] = "file",
    file_ext: Annotated[str, typer.Option("--file-ext", help="File extension for file materializer")] = ".py",
    concurrency: Annotated[Optional[int], typer.Option("--concurrency", help="Max concurrent agents")] = None,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Show agent output")] = False,
    memory: Annotated[bool, typer.Option("--memory/--no-memory", help="Enable shared memory bank — agents record and read observations, hypotheses, and learnings")] = False,
    api_key: Annotated[Optional[str], typer.Option(envvar="OPENROUTER_API_KEY", help="OpenRouter API key")] = None,
) -> None:
    """Launch concurrent agents that iteratively produce and improve solutions."""
    from fanout.db.models import Run
    from fanout.launch import launch as do_launch

    store = _get_store()

    # Resolve models
    models = model or ["openai/gpt-4o-mini"]
    if model_set:
        ms = load_model_sets().get(model_set)
        if ms:
            models = [e.model for e in ms.models]

    run = Run(prompt=prompt)
    store.save_run(run)
    console.print(f"[bold green]Created run:[/] {run.id}")
    console.print(f"Launching {n_agents} agent(s) with max {max_steps} steps each")
    if memory:
        console.print("[dim]Shared memory bank: enabled[/]")

    solutions = do_launch(
        prompt=prompt,
        models=models,
        store=store,
        run_id=run.id,
        n_agents=n_agents,
        max_steps=max_steps,
        eval_script=eval_script,
        materializer=materializer,
        file_ext=file_ext,
        concurrency=concurrency,
        verbose=verbose,
        api_key=api_key,
        use_memory=memory,
    )

    console.print(f"\n[bold green]Done.[/] {len(solutions)} solution(s) produced")

    # Show results
    scored = store.get_solutions_with_scores(run.id)
    if scored:
        table = Table(title=f"Solutions (run {run.id})")
        table.add_column("ID", style="cyan")
        table.add_column("Model")
        table.add_column("Score", justify="right", style="bold")
        table.add_column("Iteration", justify="right")
        table.add_column("Output (preview)")
        for s in scored:
            extracted = extract_solution(s.solution.output)
            iteration = s.solution.metadata.get("iteration", "?")
            table.add_row(
                s.solution.id,
                s.solution.model,
                f"{s.aggregate_score:.3f}",
                str(iteration),
                extracted[:80] + ("..." if len(extracted) > 80 else ""),
            )
        console.print(table)

    console.print(f"Run ID: {run.id}")


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
        extracted = extract_solution(s.solution.output)
        table.add_row(
            s.solution.id, s.solution.model, f"{s.aggregate_score:.3f}",
            extracted[:80] + ("..." if len(extracted) > 80 else ""),
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
    max_tokens: Annotated[int, typer.Option(help="Max tokens per response")] = 16384,
    reference: Annotated[Optional[str], typer.Option(help="Reference answer for accuracy evaluator")] = None,
    eval_script: Annotated[Optional[str], typer.Option("--eval-script", help="Path to eval script (implies -e script)")] = None,
    materializer: Annotated[str, typer.Option("--materializer", help="Materializer name")] = "file",
    file_ext: Annotated[str, typer.Option("--file-ext", help="File extension for file materializer")] = ".py",
    k_agg: Annotated[int, typer.Option("--k-agg", help="Number of parent solutions per aggregation prompt (RSA)")] = 6,
    solution_format: Annotated[str, typer.Option("--solution-format", help="Solution format: code, diff, raw")] = "code",
    eval_concurrency: Annotated[int, typer.Option("-p", "--eval-concurrency", help="Max parallel evaluations")] = 1,
    eval_timeout: Annotated[int, typer.Option("--eval-timeout", help="Timeout per evaluation in seconds")] = 60,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Show per-solution details with syntax-highlighted previews")] = False,
    full: Annotated[bool, typer.Option("--full", help="Show full solutions (not truncated)")] = False,
    api_key: Annotated[Optional[str], typer.Option(envvar="OPENROUTER_API_KEY")] = None,
    mode: Annotated[str, typer.Option("--mode", help="Workflow mode: sample or agent")] = "sample",
    n_agents: Annotated[int, typer.Option("--n-agents", help="Number of agents for agent mode")] = 3,
    max_steps: Annotated[int, typer.Option("--max-steps", help="Max steps per agent")] = 10,
    memory: Annotated[bool, typer.Option("--memory/--no-memory", help="Enable shared memory bank — agents record and read observations, hypotheses, and learnings; sample workflow injects round learnings into subsequent prompts")] = False,
) -> None:
    """Full workflow: sample → evaluate → select × N rounds, or agent mode."""
    from fanout.workflow import SampleWorkflow, LaunchWorkflow

    store = _get_store()
    models = model or ["openai/gpt-4o-mini"]
    evaluator_names = evaluator or ["latency", "cost"]

    # If --eval-script provided, add script evaluator
    if eval_script:
        if "script" not in evaluator_names:
            evaluator_names.append("script")

    eval_context: dict[str, Any] = {}
    if reference:
        eval_context["reference"] = reference
    if eval_script:
        eval_context["eval_script"] = eval_script
        eval_context["materializer"] = materializer
        eval_context["file_extension"] = file_ext
    eval_context["eval_timeout"] = eval_timeout

    lexer = _ext_to_lexer(file_ext)

    if memory:
        console.print("[dim]Shared memory bank: enabled[/]")

    if mode == "agent":
        wf = LaunchWorkflow()
        console.print(f"[bold green]Agent mode[/] — {n_agents} agent(s), max {max_steps} steps, strategy: {strategy}")
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
            eval_script=eval_script,
            eval_context=eval_context or None,
            evaluator_names=evaluator_names,
            api_key=api_key,
            store=store,
            verbose=verbose,
            full=full,
            console=console,
            syntax_lang=lexer,
            use_memory=memory,
        )
    else:
        wf = SampleWorkflow()
        if model_set:
            console.print(f"[bold green]Sample mode[/] — {rounds} round(s), model set: {model_set} (n={n_samples})")
        else:
            console.print(f"[bold green]Sample mode[/] — {rounds} round(s), {len(models)} model(s), n={n_samples}")
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
            eval_script=eval_script,
            eval_context=eval_context or None,
            evaluator_names=evaluator_names,
            eval_concurrency=eval_concurrency,
            api_key=api_key,
            store=store,
            verbose=verbose,
            full=full,
            console=console,
            syntax_lang=lexer,
            use_memory=memory,
        )

    # Show final results
    scored = store.get_solutions_with_scores(result.run_id)
    if scored:
        table = Table(title=f"Top solutions (run {result.run_id[:8]})")
        table.add_column("Rank", justify="right")
        table.add_column("Model")
        table.add_column("Score", justify="right", style="bold green")
        table.add_column("Output (preview)")
        for i, s in enumerate(scored[:5], 1):
            extracted = extract_solution(s.solution.output)
            table.add_row(
                str(i), s.solution.model, f"{s.aggregate_score:.4f}",
                extracted[:80] + ("..." if len(extracted) > 80 else ""),
            )
        console.print(table)

    console.print(f"\n[bold green]Done.[/] Run ID: {result.run_id} best={result.best_score:.4f}")


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
            extracted = extract_solution(s.solution.output)
            table.add_row(
                s.solution.id, str(s.solution.round_num), s.solution.model,
                f"{s.aggregate_score:.3f}",
                extracted[:60] + ("..." if len(extracted) > 60 else ""),
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
