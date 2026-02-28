"""Tests for EvaluationCache and its integration with evaluate_solutions."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from fanout.channels.memory import MemoryChannel
from fanout.db.models import Evaluation, Run, Solution
from fanout.evaluate import EvaluationCache, evaluate_solutions
from fanout.store import Store


# ── Helpers ──────────────────────────────────────────────


def _make_store() -> Store:
    return Store(channel=MemoryChannel())


def _make_solution(output: str = "print('hello')", run_id: str = "run") -> Solution:
    return Solution(run_id=run_id, model="test-model", output=output)


# ── EvaluationCache unit tests ───────────────────────────


class TestEvaluationCache:
    def test_miss_on_empty_cache(self):
        cache = EvaluationCache()
        sol = _make_solution()
        assert cache.get(sol, "script") is None

    def test_hit_after_put(self):
        cache = EvaluationCache()
        sol = _make_solution()
        ev = Evaluation(solution_id=sol.id, evaluator="script", score=0.9)
        cache.put(sol, "script", ev)
        assert cache.get(sol, "script") is ev

    def test_different_content_is_miss(self):
        cache = EvaluationCache()
        sol_a = _make_solution("output A")
        sol_b = _make_solution("output B")
        ev = Evaluation(solution_id=sol_a.id, evaluator="script", score=0.8)
        cache.put(sol_a, "script", ev)
        assert cache.get(sol_b, "script") is None

    def test_different_evaluator_is_miss(self):
        cache = EvaluationCache()
        sol = _make_solution()
        ev = Evaluation(solution_id=sol.id, evaluator="script", score=0.8)
        cache.put(sol, "script", ev)
        assert cache.get(sol, "latency") is None

    def test_same_content_different_id_is_hit(self):
        """Two Solution objects with identical output text share a cache entry."""
        cache = EvaluationCache()
        sol_a = _make_solution("same output")
        sol_b = _make_solution("same output")
        assert sol_a.id != sol_b.id  # different IDs
        ev = Evaluation(solution_id=sol_a.id, evaluator="script", score=0.7)
        cache.put(sol_a, "script", ev)
        hit = cache.get(sol_b, "script")
        assert hit is not None
        assert hit.score == 0.7

    def test_hit_miss_counters(self):
        cache = EvaluationCache()
        sol = _make_solution()
        ev = Evaluation(solution_id=sol.id, evaluator="script", score=0.5)
        cache.put(sol, "script", ev)

        cache.get(sol, "script")      # hit
        cache.get(sol, "latency")     # miss
        cache.get(sol, "script")      # hit

        assert cache.hits == 2
        assert cache.misses == 1

    def test_hit_rate(self):
        cache = EvaluationCache()
        sol = _make_solution()
        ev = Evaluation(solution_id=sol.id, evaluator="script", score=0.5)
        cache.put(sol, "script", ev)
        cache.get(sol, "script")   # hit
        cache.get(sol, "script")   # hit
        cache.get(sol, "latency")  # miss
        assert cache.hit_rate == pytest.approx(2 / 3)

    def test_len(self):
        cache = EvaluationCache()
        sol = _make_solution()
        assert len(cache) == 0
        ev = Evaluation(solution_id=sol.id, evaluator="script", score=0.5)
        cache.put(sol, "script", ev)
        assert len(cache) == 1

    def test_zero_hit_rate_on_empty(self):
        cache = EvaluationCache()
        assert cache.hit_rate == 0.0


# ── Integration: evaluate_solutions with cache ───────────


class TestEvaluateSolutionsCache:
    def test_cache_hit_skips_evaluator(self):
        """When a cache hit exists, the underlying evaluator must not be called."""
        store = _make_store()
        run = Run(prompt="test")
        store.save_run(run)
        sol = Solution(run_id=run.id, model="m", output="cached output")
        store.save_solution(sol)

        cache = EvaluationCache()
        # Pre-populate the cache
        cached_ev = Evaluation(
            solution_id=sol.id, evaluator="script", score=0.99, raw_score=0.99
        )
        cache.put(sol, "script", cached_ev)

        call_count = 0

        async def fake_evaluate(solution, context):
            nonlocal call_count
            call_count += 1
            return {"score": 0.0, "raw_score": 0.0, "details": {}}

        with patch(
            "fanout.evaluators.script.ScriptEvaluator.evaluate",
            new=AsyncMock(side_effect=fake_evaluate),
        ):
            evals = evaluate_solutions(
                [sol], ["script"], store, {"eval_script": "/bin/true"}, cache=cache
            )

        assert call_count == 0, "Evaluator should not be called on cache hit"
        assert evals[0].score == pytest.approx(0.99)

    def test_cache_miss_calls_evaluator_and_populates_cache(self):
        """On a miss the evaluator runs and the result is cached for next time."""
        store = _make_store()
        run = Run(prompt="test")
        store.save_run(run)
        sol = Solution(run_id=run.id, model="m", output="uncached output")
        store.save_solution(sol)

        cache = EvaluationCache()
        assert cache.get(sol, "latency") is None

        evals = evaluate_solutions([sol], ["latency"], store, cache=cache)

        assert len(evals) == 1
        assert cache.get(sol, "latency") is not None

    def test_no_cache_behaves_as_before(self):
        """Passing cache=None keeps the original behaviour (no errors)."""
        store = _make_store()
        run = Run(prompt="test")
        store.save_run(run)
        sol = Solution(run_id=run.id, model="m", output="x")
        store.save_solution(sol)

        evals = evaluate_solutions([sol], ["latency"], store, cache=None)
        assert len(evals) == 1

    def test_cached_evaluation_relinked_to_current_solution_id(self):
        """Cache hit rebases the evaluation to the current solution's ID."""
        store = _make_store()
        run = Run(prompt="test")
        store.save_run(run)

        sol_a = Solution(run_id=run.id, model="m", output="same text")
        sol_b = Solution(run_id=run.id, model="m", output="same text")
        store.save_solution(sol_a)
        store.save_solution(sol_b)

        cache = EvaluationCache()
        # Evaluate sol_a to populate the cache
        evaluate_solutions([sol_a], ["latency"], store, cache=cache)

        # Evaluate sol_b — should hit the cache, rebased to sol_b.id
        evals_b = evaluate_solutions([sol_b], ["latency"], store, cache=cache)
        assert evals_b[0].solution_id == sol_b.id
