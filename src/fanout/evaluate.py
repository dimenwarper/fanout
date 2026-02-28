"""Evaluation orchestration — run evaluators on solutions and persist scores."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from fanout.db.models import Evaluation, Solution
from fanout.evaluators.base import BaseEvaluator, get_evaluator
from fanout.store import Store

DEFAULT_EVAL_CONCURRENCY = 1


# ── Evaluation cache ─────────────────────────────────────


class EvaluationCache:
    """Content-addressed cache mapping (output_hash, evaluator) → Evaluation.

    Inspired by GEPA's EvaluationCache which caches (candidate_hash, example_id)
    pairs to avoid re-running expensive evaluations on identical outputs.

    In fanout's setting the "example" dimension doesn't exist (we evaluate each
    solution once against the task), so the key is simply the SHA-256 of the
    solution output text plus the evaluator name.  This pays off when:

    - A parent solution is carried unchanged into the next round's candidate pool.
    - Multiple runs share the same store and re-evaluate identical solutions.
    - An eval script is expensive (e.g. compiling + running code).

    The cache is intentionally *in-process only* (not persisted to Redis/SQLite)
    so it never returns stale results across restarts.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], Evaluation] = {}
        self.hits: int = 0
        self.misses: int = 0

    @staticmethod
    def _key(solution: Solution, evaluator_name: str) -> tuple[str, str]:
        content_hash = hashlib.sha256(solution.output.encode()).hexdigest()
        return (content_hash, evaluator_name)

    def get(self, solution: Solution, evaluator_name: str) -> Evaluation | None:
        """Return a cached Evaluation, or None on a cache miss."""
        entry = self._cache.get(self._key(solution, evaluator_name))
        if entry is not None:
            self.hits += 1
        else:
            self.misses += 1
        return entry

    def put(self, solution: Solution, evaluator_name: str, evaluation: Evaluation) -> None:
        """Store an Evaluation in the cache."""
        self._cache[self._key(solution, evaluator_name)] = evaluation

    def __len__(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ── Core evaluation logic ────────────────────────────────


async def _eval_one(
    ev: BaseEvaluator,
    sol: Solution,
    context: dict[str, Any] | None,
    store: Store,
    sem: asyncio.Semaphore,
    cache: EvaluationCache | None = None,
) -> Evaluation:
    # Return cached result immediately — no semaphore needed, no I/O
    if cache is not None:
        cached = cache.get(sol, ev.name)
        if cached is not None:
            # Re-link the cached evaluation to the current solution so IDs
            # stay consistent with what's saved in the store this round.
            rebased = cached.model_copy(update={"solution_id": sol.id})
            store.save_evaluation(rebased)
            return rebased

    async with sem:
        result = await ev.evaluate(sol, context)
    evaluation = ev.to_evaluation(sol, result)
    store.save_evaluation(evaluation)

    if cache is not None:
        cache.put(sol, ev.name, evaluation)

    return evaluation


async def evaluate_solutions_async(
    solutions: list[Solution],
    evaluator_names: list[str],
    store: Store,
    context: dict[str, Any] | None = None,
    concurrency: int = DEFAULT_EVAL_CONCURRENCY,
    cache: EvaluationCache | None = None,
) -> list[Evaluation]:
    """Run all named evaluators on all solutions and persist results."""
    evaluators: list[BaseEvaluator] = [get_evaluator(name) for name in evaluator_names]
    sem = asyncio.Semaphore(concurrency)

    # Build tasks in solution-major order so results stay aligned with solutions
    tasks = [
        _eval_one(ev, sol, context, store, sem, cache)
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
    cache: EvaluationCache | None = None,
) -> list[Evaluation]:
    """Synchronous wrapper around evaluate_solutions_async."""
    return asyncio.run(evaluate_solutions_async(
        solutions, evaluator_names, store, context, concurrency, cache,
    ))
