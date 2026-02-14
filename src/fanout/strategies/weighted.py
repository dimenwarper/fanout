"""Weighted random selection — probability proportional to score."""

from __future__ import annotations

import random
from typing import Any

from fanout.db.models import SolutionWithScores
from fanout.strategies.base import BaseStrategy, register_strategy


@register_strategy
class WeightedStrategy(BaseStrategy):
    name = "weighted"
    description = "Select solutions with probability proportional to their score"

    def select(self, candidates: list[SolutionWithScores], **kwargs: Any) -> list[SolutionWithScores]:
        k = kwargs.get("k", 3)
        if not candidates:
            return []
        k = min(k, len(candidates))
        weights = [max(c.aggregate_score, 1e-6) for c in candidates]
        selected: list[SolutionWithScores] = []
        pool = list(zip(candidates, weights))
        for _ in range(k):
            if not pool:
                break
            cands, ws = zip(*pool)
            choice = random.choices(list(cands), weights=list(ws), k=1)[0]
            selected.append(choice)
            pool = [(c, w) for c, w in pool if c is not choice]
        return selected
