"""Tests for the Pareto-front selection strategy."""

from __future__ import annotations

import pytest

from fanout.db.models import Evaluation, Solution, SolutionWithScores
from fanout.strategies.pareto import ParetoStrategy, pareto_front, _dominates


# ── Helpers ──────────────────────────────────────────────


def _make_candidate(scores: dict[str, float], idx: int = 0) -> SolutionWithScores:
    sol = Solution(run_id="run", model=f"model-{idx}", output="x")
    evals = [
        Evaluation(solution_id=sol.id, evaluator=name, score=score)
        for name, score in scores.items()
    ]
    aggregate = sum(scores.values()) / len(scores)
    return SolutionWithScores(solution=sol, evaluations=evals, aggregate_score=aggregate)


# ── _dominates ───────────────────────────────────────────


class TestDominates:
    def test_b_dominates_a_when_strictly_better_on_one(self):
        # b=[0.8,0.8] dominates a=[0.8,0.6]: equal on dim-0, better on dim-1
        assert _dominates([0.8, 0.6], [0.8, 0.8]) is True

    def test_a_not_dominated_when_better_on_one_dim(self):
        # a=[0.9,0.5] vs b=[0.7,0.9]: neither dominates
        assert _dominates([0.9, 0.5], [0.7, 0.9]) is False

    def test_equal_scores_not_dominated(self):
        assert _dominates([0.5, 0.5], [0.5, 0.5]) is False

    def test_different_length_not_dominated(self):
        assert _dominates([0.5], [0.5, 0.5]) is False


# ── pareto_front ─────────────────────────────────────────


class TestParetoFront:
    def test_single_candidate_always_on_front(self):
        c = _make_candidate({"a": 0.5, "b": 0.5})
        assert pareto_front([c]) == [c]

    def test_dominated_candidate_excluded(self):
        # c1 is strictly dominated by c2 on both evaluators
        c1 = _make_candidate({"latency": 0.4, "accuracy": 0.4}, idx=0)
        c2 = _make_candidate({"latency": 0.8, "accuracy": 0.8}, idx=1)
        front = pareto_front([c1, c2])
        assert c2 in front
        assert c1 not in front

    def test_trade_off_candidates_both_on_front(self):
        # c1 wins on latency, c2 wins on accuracy — neither dominates
        c1 = _make_candidate({"latency": 0.9, "accuracy": 0.4}, idx=0)
        c2 = _make_candidate({"latency": 0.4, "accuracy": 0.9}, idx=1)
        front = pareto_front([c1, c2])
        assert c1 in front
        assert c2 in front

    def test_three_way_with_one_dominated(self):
        c1 = _make_candidate({"a": 0.9, "b": 0.3}, idx=0)   # front: wins on a
        c2 = _make_candidate({"a": 0.3, "b": 0.9}, idx=1)   # front: wins on b
        c3 = _make_candidate({"a": 0.5, "b": 0.5}, idx=2)   # dominated by c1+c2? no — c1 wins a, c2 wins b
        # c3 is not dominated by c1 (c1 loses on b) nor by c2 (c2 loses on a)
        front = pareto_front([c1, c2, c3])
        assert c1 in front
        assert c2 in front
        assert c3 in front  # c3 is not dominated

    def test_clearly_dominated_middle_candidate(self):
        c1 = _make_candidate({"a": 0.9, "b": 0.9}, idx=0)  # dominates everyone
        c2 = _make_candidate({"a": 0.5, "b": 0.5}, idx=1)  # dominated
        c3 = _make_candidate({"a": 0.1, "b": 0.1}, idx=2)  # dominated
        front = pareto_front([c1, c2, c3])
        assert front == [c1]


# ── ParetoStrategy ───────────────────────────────────────


class TestParetoStrategy:
    def test_returns_k_candidates(self):
        candidates = [
            _make_candidate({"latency": 0.9, "accuracy": 0.4}, idx=0),
            _make_candidate({"latency": 0.4, "accuracy": 0.9}, idx=1),
            _make_candidate({"latency": 0.6, "accuracy": 0.6}, idx=2),
            _make_candidate({"latency": 0.2, "accuracy": 0.2}, idx=3),
        ]
        strategy = ParetoStrategy()
        selected = strategy.select(candidates, k=2)
        assert len(selected) == 2

    def test_front_members_preferred_over_non_front(self):
        # c_dominated has high aggregate but is dominated
        c_front1 = _make_candidate({"lat": 0.9, "acc": 0.3}, idx=0)  # front
        c_front2 = _make_candidate({"lat": 0.3, "acc": 0.9}, idx=1)  # front
        c_dominated = _make_candidate({"lat": 0.5, "acc": 0.5}, idx=2)  # dominated by neither above actually
        # Actually c_dominated here is NOT dominated by c_front1 (c_front1 loses on acc)
        # Let's make a clearly dominated one
        c_clearly_dominated = _make_candidate({"lat": 0.2, "acc": 0.2}, idx=3)  # dominated by all

        strategy = ParetoStrategy()
        selected = strategy.select([c_front1, c_front2, c_clearly_dominated], k=2)
        ids = {s.solution.id for s in selected}
        assert c_front1.solution.id in ids
        assert c_front2.solution.id in ids
        assert c_clearly_dominated.solution.id not in ids

    def test_falls_back_to_top_k_with_single_evaluator(self):
        candidates = [
            _make_candidate({"script": s}, idx=i)
            for i, s in enumerate([0.9, 0.7, 0.5, 0.3])
        ]
        strategy = ParetoStrategy()
        selected = strategy.select(candidates, k=2)
        scores = [s.aggregate_score for s in selected]
        assert scores == [0.9, 0.7]

    def test_fills_slots_from_remainder_when_front_small(self):
        # Only one solution on front, k=3 → should still return 3
        c1 = _make_candidate({"a": 0.9, "b": 0.9}, idx=0)  # dominates all
        c2 = _make_candidate({"a": 0.5, "b": 0.5}, idx=1)
        c3 = _make_candidate({"a": 0.3, "b": 0.3}, idx=2)
        strategy = ParetoStrategy()
        selected = strategy.select([c1, c2, c3], k=3)
        assert len(selected) == 3
        assert selected[0].solution.id == c1.solution.id  # front member first

    def test_empty_candidates(self):
        assert ParetoStrategy().select([]) == []

    def test_k_capped_at_candidates(self):
        candidates = [_make_candidate({"a": 0.5, "b": 0.5}, idx=i) for i in range(2)]
        selected = ParetoStrategy().select(candidates, k=10)
        assert len(selected) == 2

    def test_registered_by_name(self):
        from fanout.strategies.base import get_strategy
        s = get_strategy("pareto")
        assert isinstance(s, ParetoStrategy)
