"""Top-K selection strategy."""

from __future__ import annotations

from typing import Any

from fanout.db.models import SolutionWithScores
from fanout.strategies.base import BaseStrategy, register_strategy


@register_strategy
class TopKStrategy(BaseStrategy):
    name = "top-k"
    description = "Select the K highest-scoring solutions"

    def select(self, candidates: list[SolutionWithScores], **kwargs: Any) -> list[SolutionWithScores]:
        k = kwargs.get("k", 3)
        sorted_candidates = sorted(candidates, key=lambda c: c.aggregate_score, reverse=True)
        return sorted_candidates[:k]
