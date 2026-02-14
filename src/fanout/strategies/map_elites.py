"""MAP-Elites strategy — best solution per behavioral dimension cell."""

from __future__ import annotations

from typing import Any

from fanout.db.models import SolutionWithScores
from fanout.strategies.base import BaseStrategy, register_strategy


@register_strategy
class MapElitesStrategy(BaseStrategy):
    name = "map-elites"
    description = "Select the best solution per behavioral dimension cell (e.g., model, length bucket)"

    def select(self, candidates: list[SolutionWithScores], **kwargs: Any) -> list[SolutionWithScores]:
        dimension = kwargs.get("dimension", "model")
        cells: dict[str, SolutionWithScores] = {}
        for c in candidates:
            key = self._cell_key(c, dimension)
            if key not in cells or c.aggregate_score > cells[key].aggregate_score:
                cells[key] = c
        return list(cells.values())

    def _cell_key(self, candidate: SolutionWithScores, dimension: str) -> str:
        if dimension == "model":
            return candidate.solution.model
        if dimension == "length":
            length = len(candidate.solution.output)
            bucket = length // 500
            return f"len_{bucket * 500}-{(bucket + 1) * 500}"
        return candidate.solution.metadata.get(dimension, "unknown")
