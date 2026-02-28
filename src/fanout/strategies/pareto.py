"""Pareto-front selection strategy.

Inspired by GEPA's ParetoCandidateSelector and Pareto-efficient evolutionary search.

When multiple evaluators are active, collapsing them into a single aggregate
score loses information — a solution that's mediocre on every metric beats one
that's excellent on most but fails one.  Pareto selection avoids this by
preserving the *non-dominated frontier*:

    Solution A dominates solution B if A is >= B on every evaluator AND
    strictly > B on at least one.

The Pareto front is the set of solutions not dominated by any other.  All
front members represent genuinely different quality trade-offs and are equally
valid parents for the next round.

Selection procedure:
  1. Compute the Pareto front across per-evaluator scores.
  2. Sort the front by aggregate score (descending) as a tiebreaker.
  3. Return up to k front members.
  4. If the front has fewer than k members, fill remaining slots from
     non-front solutions (sorted by aggregate score).
  5. Falls back to plain top-k when only one evaluator is present (the
     Pareto front degenerates to the top-1 in that case).
"""

from __future__ import annotations

from typing import Any

from fanout.db.models import SolutionWithScores
from fanout.strategies.base import BaseStrategy, build_annotated_prompt, register_strategy


def _per_evaluator_scores(candidate: SolutionWithScores) -> list[float]:
    """Return per-evaluator scores in a stable order (sorted by evaluator name)."""
    return [score for _, score in sorted(candidate.scores_by_evaluator.items())]


def _dominates(a: list[float], b: list[float]) -> bool:
    """Return True if b dominates a (b >= a on all, b > a on at least one)."""
    if len(a) != len(b):
        return False
    return all(bv >= av for av, bv in zip(a, b)) and any(bv > av for av, bv in zip(a, b))


def pareto_front(candidates: list[SolutionWithScores]) -> list[SolutionWithScores]:
    """Return the non-dominated subset of candidates.

    A candidate is on the front if no other candidate dominates it across
    all per-evaluator scores.  With a single evaluator this returns only the
    single highest-scoring solution (use top-k instead in that case).
    """
    scores = [_per_evaluator_scores(c) for c in candidates]
    front: list[SolutionWithScores] = []
    for i, c in enumerate(candidates):
        dominated = any(
            _dominates(scores[i], scores[j])
            for j in range(len(candidates))
            if j != i
        )
        if not dominated:
            front.append(c)
    return front


@register_strategy
class ParetoStrategy(BaseStrategy):
    """Pareto-front selection across multiple evaluator objectives.

    Selects the non-dominated frontier first, then fills remaining slots
    with the best non-front solutions by aggregate score.  Falls back to
    top-k when only one evaluator is present.
    """

    name = "pareto"
    description = (
        "Pareto-front selection: preserve non-dominated solutions across all "
        "evaluator objectives, then fill remaining slots by aggregate score "
        "(inspired by GEPA's Pareto-efficient evolutionary search)"
    )

    def select(
        self,
        candidates: list[SolutionWithScores],
        *,
        k: int = 3,
        **kwargs: Any,
    ) -> list[SolutionWithScores]:
        if not candidates:
            return []

        k = min(k, len(candidates))

        # With a single evaluator the Pareto front degenerates — use top-k
        n_evaluators = max(len(c.evaluations) for c in candidates)
        if n_evaluators <= 1:
            return sorted(candidates, key=lambda c: c.aggregate_score, reverse=True)[:k]

        front = pareto_front(candidates)
        front_sorted = sorted(front, key=lambda c: c.aggregate_score, reverse=True)

        if len(front_sorted) >= k:
            return front_sorted[:k]

        # Fill remaining slots from non-front candidates
        front_ids = {id(c) for c in front_sorted}
        remainder = sorted(
            [c for c in candidates if id(c) not in front_ids],
            key=lambda c: c.aggregate_score,
            reverse=True,
        )
        return front_sorted + remainder[: k - len(front_sorted)]

    def build_prompts(
        self,
        original_prompt: str,
        selected: list[SolutionWithScores],
        round_num: int,
        n_samples: int,
        **kwargs: Any,
    ) -> str | list[str]:
        if round_num == 0 or not selected:
            return original_prompt
        return build_annotated_prompt(
            original_prompt,
            selected,
            instruction=(
                "These solutions represent diverse trade-offs across multiple objectives. "
                "Produce an improved solution that balances all objectives well, "
                "addressing the specific weaknesses shown in the error output."
            ),
        )
