"""Island strategy — subpopulation evolution with migration."""

from __future__ import annotations

import random
from typing import Any

from fanout.db.models import SolutionWithScores
from fanout.strategies.base import BaseStrategy, register_strategy


@register_strategy
class IslandStrategy(BaseStrategy):
    name = "island"
    description = "Evolve subpopulations per model with periodic migration of top solutions"

    def select(self, candidates: list[SolutionWithScores], **kwargs: Any) -> list[SolutionWithScores]:
        k_per_island = kwargs.get("k_per_island", 2)
        migration_rate = kwargs.get("migration_rate", 0.2)

        # Group by model (island)
        islands: dict[str, list[SolutionWithScores]] = {}
        for c in candidates:
            islands.setdefault(c.solution.model, []).append(c)

        selected: list[SolutionWithScores] = []
        island_bests: list[SolutionWithScores] = []

        for _model, members in islands.items():
            members.sort(key=lambda c: c.aggregate_score, reverse=True)
            top = members[:k_per_island]
            selected.extend(top)
            if top:
                island_bests.append(top[0])

        # Migration: with some probability, share top solutions across islands
        if len(island_bests) > 1 and random.random() < migration_rate:
            migrant = max(island_bests, key=lambda c: c.aggregate_score)
            if migrant not in selected:
                selected.append(migrant)

        return selected
