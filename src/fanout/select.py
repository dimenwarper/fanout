"""Selection orchestration — pick the best solutions for the next round."""

from __future__ import annotations

from typing import Any

from fanout.db.models import SolutionWithScores
from fanout.strategies.base import get_strategy
from fanout.store import Store


def select_solutions(
    run_id: str,
    round_num: int,
    strategy_name: str,
    store: Store,
    **kwargs: Any,
) -> list[SolutionWithScores]:
    """Score, rank, and select solutions from a round."""
    candidates = store.get_solutions_with_scores(run_id, round_num)
    strategy = get_strategy(strategy_name)
    return strategy.select(candidates, **kwargs)
