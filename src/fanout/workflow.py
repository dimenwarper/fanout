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

from dataclasses import dataclass, field
from typing import Any, Callable

from fanout.db.models import Evaluation, Run, Solution, SolutionWithScores
from fanout.evaluate import evaluate_solutions
from fanout.providers.openrouter import SamplingConfig
from fanout.sample import sample as do_sample
from fanout.select import select_solutions
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
        )

        for rnd in range(rounds):
            ctx.round_num = rnd
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
