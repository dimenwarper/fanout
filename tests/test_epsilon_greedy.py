"""Tests for the epsilon-greedy selection strategy."""

from __future__ import annotations

import pytest

from fanout.db.models import Evaluation, Solution, SolutionWithScores
from fanout.strategies.epsilon_greedy import EpsilonGreedyStrategy


def _make_candidate(score: float, idx: int = 0) -> SolutionWithScores:
    sol = Solution(run_id="run", model=f"model-{idx}", output="x")
    ev = Evaluation(solution_id=sol.id, evaluator="script", score=score)
    return SolutionWithScores(solution=sol, evaluations=[ev], aggregate_score=score)


class TestEpsilonGreedyStrategy:
    def test_returns_k_candidates(self):
        candidates = [_make_candidate(s, i) for i, s in enumerate([0.9, 0.7, 0.5, 0.3])]
        strategy = EpsilonGreedyStrategy()
        selected = strategy.select(candidates, k=2, epsilon=0.0)
        assert len(selected) == 2

    def test_pure_greedy_picks_top_k(self):
        """epsilon=0 → always pick best remaining (deterministic top-k)."""
        candidates = [_make_candidate(s, i) for i, s in enumerate([0.9, 0.7, 0.5, 0.3])]
        strategy = EpsilonGreedyStrategy()
        selected = strategy.select(candidates, k=2, epsilon=0.0)
        scores = [s.aggregate_score for s in selected]
        assert scores == [0.9, 0.7]

    def test_no_duplicates_in_selection(self):
        candidates = [_make_candidate(s, i) for i, s in enumerate([0.9, 0.8, 0.7])]
        strategy = EpsilonGreedyStrategy()
        for _ in range(20):
            selected = strategy.select(candidates, k=3, epsilon=0.5)
            ids = [s.solution.id for s in selected]
            assert len(ids) == len(set(ids)), "Duplicates found in selection"

    def test_k_capped_at_candidates(self):
        candidates = [_make_candidate(0.5, i) for i in range(2)]
        strategy = EpsilonGreedyStrategy()
        selected = strategy.select(candidates, k=10, epsilon=0.0)
        assert len(selected) == 2

    def test_empty_candidates(self):
        strategy = EpsilonGreedyStrategy()
        assert strategy.select([], k=3) == []

    def test_registered_by_name(self):
        from fanout.strategies.base import get_strategy
        s = get_strategy("epsilon-greedy")
        assert isinstance(s, EpsilonGreedyStrategy)
