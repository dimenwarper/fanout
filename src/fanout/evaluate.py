"""Evaluation orchestration — run evaluators on solutions and persist scores."""

from __future__ import annotations

import asyncio
from typing import Any

from fanout.db.models import Evaluation, Solution
from fanout.evaluators.base import BaseEvaluator, get_evaluator
from fanout.store import Store


async def evaluate_solutions_async(
    solutions: list[Solution],
    evaluator_names: list[str],
    store: Store,
    context: dict[str, Any] | None = None,
) -> list[Evaluation]:
    """Run all named evaluators on all solutions and persist results."""
    evaluators: list[BaseEvaluator] = [get_evaluator(name) for name in evaluator_names]
    all_evals: list[Evaluation] = []

    for sol in solutions:
        for ev in evaluators:
            result = await ev.evaluate(sol, context)
            evaluation = ev.to_evaluation(sol, result)
            store.save_evaluation(evaluation)
            all_evals.append(evaluation)

    return all_evals


def evaluate_solutions(
    solutions: list[Solution],
    evaluator_names: list[str],
    store: Store,
    context: dict[str, Any] | None = None,
) -> list[Evaluation]:
    """Synchronous wrapper around evaluate_solutions_async."""
    return asyncio.run(evaluate_solutions_async(solutions, evaluator_names, store, context))
