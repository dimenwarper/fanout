"""Evaluation orchestration — run evaluators on solutions and persist scores."""

from __future__ import annotations

import asyncio
from typing import Any

from fanout.db.models import Evaluation, Solution
from fanout.evaluators.base import BaseEvaluator, get_evaluator
from fanout.store import Store

DEFAULT_EVAL_CONCURRENCY = 1


async def _eval_one(
    ev: BaseEvaluator,
    sol: Solution,
    context: dict[str, Any] | None,
    store: Store,
    sem: asyncio.Semaphore,
) -> Evaluation:
    async with sem:
        result = await ev.evaluate(sol, context)
    evaluation = ev.to_evaluation(sol, result)
    store.save_evaluation(evaluation)
    return evaluation


async def evaluate_solutions_async(
    solutions: list[Solution],
    evaluator_names: list[str],
    store: Store,
    context: dict[str, Any] | None = None,
    concurrency: int = DEFAULT_EVAL_CONCURRENCY,
) -> list[Evaluation]:
    """Run all named evaluators on all solutions and persist results."""
    evaluators: list[BaseEvaluator] = [get_evaluator(name) for name in evaluator_names]
    sem = asyncio.Semaphore(concurrency)

    # Build tasks in solution-major order so results stay aligned with solutions
    tasks = [
        _eval_one(ev, sol, context, store, sem)
        for sol in solutions
        for ev in evaluators
    ]
    return list(await asyncio.gather(*tasks))


def evaluate_solutions(
    solutions: list[Solution],
    evaluator_names: list[str],
    store: Store,
    context: dict[str, Any] | None = None,
    concurrency: int = DEFAULT_EVAL_CONCURRENCY,
) -> list[Evaluation]:
    """Synchronous wrapper around evaluate_solutions_async."""
    return asyncio.run(evaluate_solutions_async(
        solutions, evaluator_names, store, context, concurrency,
    ))
