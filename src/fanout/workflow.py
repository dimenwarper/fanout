"""Composable workflow pipelines from fanout primitives.

A Workflow runs a list of step functions in a loop for N rounds.
Each step is ``(ctx: WorkflowContext) -> None`` and reads/writes shared context.
Built-in steps wrap the existing sample / evaluate / select primitives.

Usage::

    from fanout.workflow import Workflow, sample_step, evaluate_step, select_step, evolve_step

    wf = Workflow(steps=[sample_step, evaluate_step, select_step, evolve_step])
    result = wf.run(
        prompt=prompt, models=models, rounds=3,
        eval_script=str(eval_wrapper), strategy="top-k", k=3,
    )
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from rich.console import Console
from rich.syntax import Syntax

from fanout.db.models import Evaluation, Run, Solution, SolutionWithScores
from fanout.evaluate import evaluate_solutions
from fanout.providers.openrouter import SamplingConfig
from fanout.sample import sample as do_sample
from fanout.select import select_solutions
from fanout.solution_format import extract_solution, get_format
from fanout.store import Store
from fanout.strategies.base import BaseStrategy, get_strategy


@dataclass
class WorkflowContext:
    """Shared mutable state threaded through every step."""

    # ── Set at init ──────────────────────────────────────
    prompt: str
    store: Store
    run: Run
    config: SamplingConfig
    eval_context: dict[str, Any]
    evaluator_names: list[str]
    strategy_name: str
    strategy_instance: BaseStrategy
    k: int
    k_agg: int
    eval_concurrency: int
    rounds: int
    api_key: str | None = None

    # ── Logging ──────────────────────────────────────────
    console: Console | None = None
    verbose: bool = False
    full: bool = False
    syntax_lang: str = "python"

    # ── Set during loop ──────────────────────────────────
    round_num: int = 0
    current_prompt: str | list[str] = ""
    parent_ids: list[str] | None = None
    solutions: list[Solution] = field(default_factory=list)
    evaluations: list[Evaluation] = field(default_factory=list)
    selected: list[SolutionWithScores] = field(default_factory=list)
    best_score: float = 0.0
    round_scores: list[float] = field(default_factory=list)
    stop: bool = False


StepFn = Callable[[WorkflowContext], None]


@dataclass
class WorkflowResult:
    """Returned by ``Workflow.run()``."""

    run_id: str
    best_score: float
    round_scores: list[float]


# ── Built-in steps ───────────────────────────────────────


def sample_step(ctx: WorkflowContext) -> None:
    """Fan out the current prompt to models and collect solutions."""
    ctx.solutions = do_sample(
        ctx.current_prompt,
        ctx.config,
        ctx.store,
        ctx.run.id,
        ctx.round_num,
        ctx.parent_ids,
        api_key=ctx.api_key,
    )
    if ctx.console:
        counts = Counter(s.model for s in ctx.solutions)
        parts = [f"{m}(x{n})" if n > 1 else m for m, n in counts.items()]
        ctx.console.print(f"sampled [{', '.join(parts)}]", end=" ")


def evaluate_step(ctx: WorkflowContext) -> None:
    """Evaluate all solutions from the current round."""
    ctx.evaluations = evaluate_solutions(
        ctx.solutions,
        ctx.evaluator_names,
        ctx.store,
        ctx.eval_context,
        concurrency=ctx.eval_concurrency,
    )


def select_step(ctx: WorkflowContext) -> None:
    """Select the best solutions and update scores."""
    ctx.selected = select_solutions(
        ctx.run.id,
        ctx.round_num,
        ctx.strategy_name,
        ctx.store,
        k=ctx.k,
    )
    top_score = ctx.selected[0].aggregate_score if ctx.selected else 0.0
    ctx.round_scores.append(top_score)
    ctx.best_score = max(ctx.best_score, top_score)

    if ctx.console:
        ctx.console.print(f"top={top_score:.4f} best={ctx.best_score:.4f}")
        if (ctx.verbose or ctx.full) and ctx.selected:
            for i, sw in enumerate(ctx.selected):
                code = extract_solution(sw.solution.output)
                if not ctx.full:
                    lines = code.splitlines()
                    if len(lines) > 20:
                        code = "\n".join(lines[:20]) + f"\n... ({len(lines) - 20} more lines)"
                ctx.console.print(f"  [{i + 1}] score={sw.aggregate_score:.4f} model={sw.solution.model}")
                ctx.console.print(Syntax(code, ctx.syntax_lang, theme="monokai", line_numbers=True))
                # Show eval details if available
                for ev in sw.evaluations:
                    d = ev.details
                    parts = [f"exit={d['exit_code']}" if "exit_code" in d else None,
                             f"stderr={d['stderr']!r}" if d.get("stderr") else None,
                             f"stdout={d['stdout']!r}" if d.get("stdout") else None]
                    detail = " ".join(p for p in parts if p)
                    if detail:
                        ctx.console.print(f"    {detail}")


def evolve_step(ctx: WorkflowContext) -> None:
    """Build prompts for the next round and advance the run."""
    ctx.store.update_run_round(ctx.run.id, ctx.round_num + 1)
    ctx.parent_ids = [s.solution.id for s in ctx.selected]

    if ctx.round_num < ctx.rounds - 1:
        ctx.current_prompt = ctx.strategy_instance.build_prompts(
            original_prompt=ctx.prompt,
            selected=ctx.selected,
            round_num=ctx.round_num,
            n_samples=ctx.config.n_samples,
            k_agg=ctx.k_agg,
        )


# ── Workflow ─────────────────────────────────────────────


class Workflow:
    """A composable pipeline that loops step functions over rounds."""

    def __init__(self, steps: list[StepFn]) -> None:
        self.steps = steps

    def run(
        self,
        prompt: str,
        *,
        models: list[str] | None = None,
        model_set: str | None = None,
        n_samples: int = 5,
        rounds: int = 3,
        strategy: str = "top-k",
        k: int = 3,
        k_agg: int = 6,
        temperature: float = 0.7,
        max_tokens: int = 16384,
        solution_format: str = "code",
        eval_script: str | None = None,
        eval_context: dict[str, Any] | None = None,
        evaluator_names: list[str] | None = None,
        eval_concurrency: int = 1,
        api_key: str | None = None,
        store: Store | None = None,
        verbose: bool = False,
        full: bool = False,
        console: Console | None = None,
        syntax_lang: str = "python",
    ) -> WorkflowResult:
        """Execute the workflow loop and return results."""
        if store is None:
            store = Store()

        run = Run(prompt=prompt, total_rounds=rounds)
        store.save_run(run)

        config = SamplingConfig(
            models=models or ["openai/gpt-4o-mini"],
            temperature=temperature,
            max_tokens=max_tokens,
            model_set=model_set,
            n_samples=n_samples,
            solution_format=solution_format,
        )

        if eval_context is None:
            eval_context = {}
        if eval_script is not None:
            eval_context.setdefault("eval_script", eval_script)
            eval_context.setdefault("materializer", "file")

        strategy_instance = get_strategy(strategy)

        # Enable console logging if verbose/full requested
        if (verbose or full) and console is None:
            console = Console()

        ctx = WorkflowContext(
            prompt=prompt,
            store=store,
            run=run,
            config=config,
            eval_context=eval_context,
            evaluator_names=evaluator_names or ["script"],
            strategy_name=strategy,
            strategy_instance=strategy_instance,
            k=k,
            k_agg=k_agg,
            eval_concurrency=eval_concurrency,
            rounds=rounds,
            api_key=api_key,
            current_prompt=prompt,
            console=console,
            verbose=verbose,
            full=full,
            syntax_lang=syntax_lang,
        )

        # Pre-loop: show system prompt + prompt suffix
        if full and console:
            fmt = get_format(solution_format)
            if fmt.system_prompt:
                console.print("[dim]System prompt:[/dim]")
                console.print(fmt.system_prompt)
            if fmt.prompt_suffix:
                console.print("[dim]Prompt suffix:[/dim]")
                console.print(fmt.prompt_suffix)
            console.print()

        for rnd in range(rounds):
            ctx.round_num = rnd

            if console:
                console.print(f"Round {rnd + 1}/{rounds}...", end=" ")

            # Pre-sample: show current prompt(s)
            if full and console:
                console.print()
                if isinstance(ctx.current_prompt, list):
                    for i, p in enumerate(ctx.current_prompt):
                        console.print(f"[dim]Prompt {i + 1}:[/dim]")
                        console.print(p)
                else:
                    console.print(f"[dim]Prompt:[/dim]")
                    console.print(ctx.current_prompt)

            for step in self.steps:
                step(ctx)
                if ctx.stop:
                    break
            if ctx.stop:
                break

        return WorkflowResult(
            run_id=run.id,
            best_score=ctx.best_score,
            round_scores=ctx.round_scores,
        )
