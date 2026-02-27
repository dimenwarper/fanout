"""Composable workflow pipelines from fanout primitives.

Two workflow classes wrap the core sample/evaluate/select/evolve primitives:

- ``SampleWorkflow`` — multi-round evolutionary loop (sample → eval → select → evolve).
- ``LaunchWorkflow`` — single-shot agent-based workflow (launch → select).

Both inherit from ``Workflow`` which provides shared setup logic.

Usage::

    from fanout.workflow import SampleWorkflow, LaunchWorkflow

    # Sample mode
    wf = SampleWorkflow()
    result = wf.run(prompt=prompt, rounds=3, strategy="top-k", k=3, ...)

    # Agent mode
    wf = LaunchWorkflow()
    result = wf.run(prompt=prompt, n_agents=3, max_steps=10, ...)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from fanout.db.models import Evaluation, Memory, Run, Solution, SolutionWithScores
from fanout.evaluate import evaluate_solutions
from fanout.launch import launch as do_launch
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
    # When True, agents share learnings via the memory bank (launch workflow),
    # and the sample workflow auto-records round learnings + injects them into
    # the next round's prompt via evolve_step.
    use_memory: bool = False

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
    store: Store | None = None


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
        ctx.console.print(f"[dim]sampled \\[{', '.join(parts)}][/]", end=" ")


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
        if ctx.full and ctx.solutions:
            for i, sol in enumerate(ctx.solutions):
                # Find matching evaluation
                evals = [e for e in ctx.evaluations if e.solution_id == sol.id]
                ev = evals[0] if evals else None
                score = ev.score if ev else 0.0
                exit_code = ev.details.get("exit_code", "?") if ev else "?"
                stderr = ev.details.get("stderr", "") if ev else ""
                stdout = ev.details.get("stdout", "") if ev else ""

                if ctx.full:
                    preview = sol.output
                else:
                    extracted = extract_solution(sol.output)
                    preview_lines = extracted[:500].splitlines()[:15]
                    preview = "\n".join(preview_lines)

                ctx.console.print(f"    [dim]Solution {i + 1} [{sol.model}] score={score:.4f} exit={exit_code}[/]")
                ctx.console.print(Syntax(preview, ctx.syntax_lang, theme="monokai", line_numbers=True, padding=(0, 2)))
                if stderr:
                    ctx.console.print(f"      [dim]stderr: {stderr}[/]")
                if score == 0.0 and stdout:
                    ctx.console.print(f"      [dim]stdout: {stdout}[/]")


_MEMORY_TYPE_STYLE: dict[str, str] = {
    "observation": "cyan",
    "hypothesis":  "yellow",
    "learning":    "green",
    "strategy":    "magenta",
}


def _print_memory_table(ctx: WorkflowContext) -> None:
    """Render the current memory bank as a Rich table to ctx.console."""
    if not (ctx.use_memory and ctx.console):
        return
    memories = ctx.store.get_memories_for_run(ctx.run.id)
    if not memories:
        return

    table = Table(show_header=True, header_style="bold", expand=True, box=None)
    table.add_column("Type",    style="bold", width=12)
    table.add_column("Agent",   style="dim",  width=22)
    table.add_column("Score",   justify="right", width=7)
    table.add_column("Content")

    for mem in memories:
        style = _MEMORY_TYPE_STYLE.get(mem.memory_type, "white")
        score_str = f"{mem.score:.3f}" if mem.score is not None else "—"
        content = mem.content if len(mem.content) <= 120 else mem.content[:117] + "..."
        table.add_row(
            f"[{style}]{mem.memory_type}[/]",
            mem.agent_id,
            score_str,
            content,
        )

    ctx.console.print(
        Panel(table, title=f"[bold]Memory Bank[/] ({len(memories)} entries)", border_style="dim blue")
    )


def memory_step(ctx: WorkflowContext) -> None:
    """Auto-record round learnings into the memory bank (sample workflow).

    Writes one ``learning`` memory for the best solution and, when there is a
    meaningful score gap, one for the weakest selected solution.  No LLM call
    is made — learnings are assembled from evaluation metadata already in the
    store.  These memories are injected into the next round's prompt by
    ``evolve_step``.
    """
    if not ctx.selected:
        return

    best = ctx.selected[0]
    extracted = extract_solution(best.solution.output)
    preview = extracted[:300] + ("..." if len(extracted) > 300 else "")

    ctx.store.save_memory(Memory(
        run_id=ctx.run.id,
        agent_id=f"workflow/round-{ctx.round_num}",
        memory_type="learning",
        content=(
            f"Round {ctx.round_num} best (score={best.aggregate_score:.3f}, "
            f"model={best.solution.model}): {preview}"
        ),
        solution_id=best.solution.id,
        score=best.aggregate_score,
    ))

    if len(ctx.selected) > 1:
        worst = ctx.selected[-1]
        gap = best.aggregate_score - worst.aggregate_score
        if gap > 0.1:
            # Include any stderr hint from the worst solution's evaluations
            error_hint = ""
            for ev in worst.evaluations:
                stderr = ev.details.get("stderr", "")
                if stderr:
                    error_hint = f"  Errors: {stderr[:120]}"
                    break
            ctx.store.save_memory(Memory(
                run_id=ctx.run.id,
                agent_id=f"workflow/round-{ctx.round_num}",
                memory_type="learning",
                content=(
                    f"Round {ctx.round_num} weakest selected "
                    f"(score={worst.aggregate_score:.3f}, model={worst.solution.model})."
                    f"{error_hint}"
                ),
                solution_id=worst.solution.id,
                score=worst.aggregate_score,
            ))

    _print_memory_table(ctx)


def evolve_step(ctx: WorkflowContext) -> None:
    """Build prompts for the next round and advance the run."""
    ctx.store.update_run_round(ctx.run.id, ctx.round_num + 1)
    ctx.parent_ids = [s.solution.id for s in ctx.selected]

    if ctx.round_num < ctx.rounds - 1:
        prompt = ctx.strategy_instance.build_prompts(
            original_prompt=ctx.prompt,
            selected=ctx.selected,
            round_num=ctx.round_num,
            n_samples=ctx.config.n_samples,
            k_agg=ctx.k_agg,
        )

        # Prepend accumulated memories when the memory bank is enabled
        if ctx.use_memory:
            memories = ctx.store.get_memories_for_run(ctx.run.id)
            if memories:
                lines = ["=== Shared Learnings from Previous Rounds ==="]
                for mem in memories:
                    score_str = f" (score={mem.score:.3f})" if mem.score is not None else ""
                    lines.append(f"[{mem.memory_type}]{score_str}: {mem.content}")
                lines.append("=== End of Learnings ===\n")
                prefix = "\n".join(lines) + "\n"
                if isinstance(prompt, list):
                    prompt = [prefix + p for p in prompt]
                else:
                    prompt = prefix + prompt

        ctx.current_prompt = prompt


def launch_step(ctx: WorkflowContext, *, n_agents: int = 3, max_steps: int = 10) -> None:
    """Launch concurrent agents that iteratively produce and improve solutions."""
    if ctx.config.model_set:
        from fanout.model_sets import get_model_set, pick_models

        ms = get_model_set(ctx.config.model_set)
        models = pick_models(ms, n_agents)
    else:
        models = ctx.config.models

    ctx.solutions = do_launch(
        prompt=ctx.prompt,
        models=models,
        store=ctx.store,
        run_id=ctx.run.id,
        n_agents=n_agents,
        max_steps=max_steps,
        eval_script=ctx.eval_context.get("eval_script"),
        materializer=ctx.eval_context.get("materializer", "file"),
        file_ext=ctx.eval_context.get("file_extension", ".py"),
        verbose=ctx.verbose,
        api_key=ctx.api_key,
        console=ctx.console,
        use_memory=ctx.use_memory,
    )
    if ctx.console:
        ctx.console.print(f"[dim]launched {n_agents} agent(s), {len(ctx.solutions)} solution(s)[/]")
    _print_memory_table(ctx)


# ── Workflow base class ──────────────────────────────────


class Workflow:
    """Base class for workflow pipelines. Subclasses override ``_execute``."""

    def _build_context(
        self,
        prompt: str,
        *,
        models: list[str] | None = None,
        model_set: str | None = None,
        n_samples: int = 5,
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
        strategy: str = "top-k",
        k: int = 3,
        k_agg: int = 6,
        rounds: int = 1,
        use_memory: bool = False,
    ) -> WorkflowContext:
        """Build shared context — called by subclass ``run()`` methods."""
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

        return WorkflowContext(
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
            use_memory=use_memory,
        )

    def _log_prompt_preview(self, ctx: WorkflowContext, solution_format: str = "code") -> None:
        """Log prompt preview if verbose/full mode is enabled."""
        if ctx.console and ctx.verbose and not ctx.full:
            ctx.console.print(f"\n  [dim]Prompt ({len(ctx.prompt)} chars):[/]")
            ctx.console.print(f"  [dim]{ctx.prompt[:200]}...[/]\n")

        if ctx.full and ctx.console:
            fmt = get_format(solution_format)
            if fmt.system_prompt:
                ctx.console.print(f"\n  [bold]System prompt:[/]")
                ctx.console.print(Syntax(fmt.system_prompt, "text", theme="monokai", padding=(0, 2)))
            if fmt.prompt_suffix:
                ctx.console.print(f"\n  [bold]Prompt suffix (appended to all user prompts):[/]")
                ctx.console.print(Syntax(fmt.prompt_suffix, "text", theme="monokai", padding=(0, 2)))

    def _execute(self, ctx: WorkflowContext) -> None:
        """Override in subclasses to implement the workflow loop."""
        raise NotImplementedError

    def run(self, prompt: str, **kwargs: Any) -> WorkflowResult:
        """Execute the workflow and return results."""
        ctx = self._build_context(prompt, **kwargs)
        self._log_prompt_preview(ctx, kwargs.get("solution_format", "code"))
        self._execute(ctx)
        return WorkflowResult(
            run_id=ctx.run.id,
            best_score=ctx.best_score,
            round_scores=ctx.round_scores,
            store=ctx.store,
        )


# ── SampleWorkflow ───────────────────────────────────────


class SampleWorkflow(Workflow):
    """Multi-round evolutionary workflow: sample → eval → select → [extra] → evolve."""

    def __init__(self, *, extra_steps: list[StepFn] | None = None) -> None:
        self.extra_steps = extra_steps or []

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
        use_memory: bool = False,
    ) -> WorkflowResult:
        """Execute the sample workflow loop and return results."""
        ctx = self._build_context(
            prompt,
            models=models, model_set=model_set, n_samples=n_samples,
            temperature=temperature, max_tokens=max_tokens,
            solution_format=solution_format, eval_script=eval_script,
            eval_context=eval_context, evaluator_names=evaluator_names,
            eval_concurrency=eval_concurrency, api_key=api_key,
            store=store, verbose=verbose, full=full, console=console,
            syntax_lang=syntax_lang, strategy=strategy, k=k, k_agg=k_agg,
            rounds=rounds, use_memory=use_memory,
        )
        self._log_prompt_preview(ctx, solution_format)
        self._execute(ctx)
        return WorkflowResult(
            run_id=ctx.run.id,
            best_score=ctx.best_score,
            round_scores=ctx.round_scores,
            store=ctx.store,
        )

    def _execute(self, ctx: WorkflowContext) -> None:
        # memory_step is inserted after select_step when the memory bank is on,
        # so learnings from each round are stored before evolve_step builds the
        # next prompt (which prepends those memories as context).
        steps: list[StepFn] = [
            sample_step, evaluate_step, select_step,
            *([memory_step] if ctx.use_memory else []),
            *self.extra_steps,
            evolve_step,
        ]

        for rnd in range(ctx.rounds):
            ctx.round_num = rnd

            if ctx.console:
                ctx.console.print(f"  [dim]Round {rnd + 1}/{ctx.rounds}...[/]", end=" ")

            # Pre-sample: show current prompt(s)
            if ctx.full and ctx.console:
                if isinstance(ctx.current_prompt, list):
                    for i, p in enumerate(ctx.current_prompt):
                        ctx.console.print(f"\n  [bold]Prompt {i + 1}/{len(ctx.current_prompt)} ({len(p)} chars):[/]")
                        ctx.console.print(Syntax(p, "text", theme="monokai", padding=(0, 2)))
                else:
                    ctx.console.print(f"\n  [bold]Prompt ({len(ctx.current_prompt)} chars):[/]")
                    ctx.console.print(Syntax(ctx.current_prompt, "text", theme="monokai", padding=(0, 2)))
                ctx.console.print()

            for step in steps:
                step(ctx)
                if ctx.stop:
                    break
            if ctx.stop:
                break


# ── LaunchWorkflow ───────────────────────────────────────


class LaunchWorkflow(Workflow):
    """Single-shot agent-based workflow: launch → select."""

    def run(
        self,
        prompt: str,
        *,
        models: list[str] | None = None,
        model_set: str | None = None,
        n_samples: int = 5,
        n_agents: int = 3,
        max_steps: int = 10,
        strategy: str = "top-k",
        k: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 16384,
        solution_format: str = "code",
        eval_script: str | None = None,
        eval_context: dict[str, Any] | None = None,
        evaluator_names: list[str] | None = None,
        api_key: str | None = None,
        store: Store | None = None,
        verbose: bool = False,
        full: bool = False,
        console: Console | None = None,
        syntax_lang: str = "python",
        use_memory: bool = False,
    ) -> WorkflowResult:
        """Execute the launch workflow and return results."""
        self._n_agents = n_agents
        self._max_steps = max_steps
        ctx = self._build_context(
            prompt,
            models=models, model_set=model_set, n_samples=n_samples,
            temperature=temperature, max_tokens=max_tokens,
            solution_format=solution_format, eval_script=eval_script,
            eval_context=eval_context, evaluator_names=evaluator_names,
            api_key=api_key, store=store, verbose=verbose, full=full,
            console=console, syntax_lang=syntax_lang, strategy=strategy,
            k=k, rounds=1, use_memory=use_memory,
        )
        self._log_prompt_preview(ctx, solution_format)
        self._execute(ctx)
        return WorkflowResult(
            run_id=ctx.run.id,
            best_score=ctx.best_score,
            round_scores=ctx.round_scores,
            store=ctx.store,
        )

    def _execute(self, ctx: WorkflowContext) -> None:
        launch_step(ctx, n_agents=self._n_agents, max_steps=self._max_steps)
        select_step(ctx)
        if ctx.use_memory:
            memory_step(ctx)
