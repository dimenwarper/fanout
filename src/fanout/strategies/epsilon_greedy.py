"""Epsilon-greedy selection strategy.

Inspired by GEPA's EpsilonGreedyCandidateSelector.

With probability ``epsilon`` a random candidate is chosen; otherwise the best
remaining candidate is taken.  Runs ``k`` rounds of selection without
replacement so the result is a diverse set of ``k`` candidates where most are
high-scoring but a few are exploratory picks.

This is a lighter-weight alternative to the darwinian strategy — useful as a
simple exploration-vs-exploitation baseline.
"""

from __future__ import annotations

import random
from typing import Any

from fanout.db.models import SolutionWithScores
from fanout.strategies.base import BaseStrategy, register_strategy


@register_strategy
class EpsilonGreedyStrategy(BaseStrategy):
    """Epsilon-greedy selection without replacement.

    Parameters (passed as kwargs from select_solutions / select_step):
      k       — number of parents to select (default 3)
      epsilon — exploration probability (default 0.1)
    """

    name = "epsilon-greedy"
    description = (
        "Epsilon-greedy selection: pick the best with probability 1−ε, "
        "a random candidate with probability ε (no replacement)"
    )

    def select(
        self,
        candidates: list[SolutionWithScores],
        *,
        k: int = 3,
        epsilon: float = 0.1,
        **kwargs: Any,
    ) -> list[SolutionWithScores]:
        if not candidates:
            return []

        k = min(k, len(candidates))
        pool = sorted(candidates, key=lambda c: c.aggregate_score, reverse=True)

        selected: list[SolutionWithScores] = []
        for _ in range(k):
            if not pool:
                break
            if random.random() < epsilon:
                # Exploration: pick a random remaining candidate
                chosen = random.choice(pool)
            else:
                # Exploitation: pick the best remaining candidate
                chosen = pool[0]
            selected.append(chosen)
            pool.remove(chosen)

        return selected
