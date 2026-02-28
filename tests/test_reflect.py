"""Tests for reflective mutation — reflect() and reflect_step()."""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from fanout.channels.memory import MemoryChannel
from fanout.db.models import Evaluation, Run, Solution, SolutionWithScores
from fanout.providers.openrouter import SamplingConfig
from fanout.reflect import _build_entries, reflect
from fanout.store import Store
from fanout.strategies.base import get_strategy
from fanout.workflow import WorkflowContext, reflect_step


# ── Helpers ──────────────────────────────────────────────


def _make_store() -> Store:
    return Store(channel=MemoryChannel())


def _make_candidate(
    score: float,
    model: str = "test-model",
    output: str = "print('hello')",
    stderr: str = "",
    exit_code: int = 0,
) -> SolutionWithScores:
    sol = Solution(run_id="run", model=model, output=output)
    ev = Evaluation(
        solution_id=sol.id,
        evaluator="script",
        score=score,
        raw_score=score,
        details={"stderr": stderr, "exit_code": exit_code, "stdout": ""},
    )
    return SolutionWithScores(solution=sol, evaluations=[ev], aggregate_score=score)


def _make_ctx(store: Store | None = None, use_reflection: bool = True, rounds: int = 3, **overrides) -> WorkflowContext:
    store = store or _make_store()
    run = Run(prompt="test prompt", total_rounds=rounds)
    store.save_run(run)
    defaults = dict(
        prompt="test prompt",
        store=store,
        run=run,
        config=SamplingConfig(models=["test-model"], n_samples=2),
        eval_context={},
        evaluator_names=["script"],
        strategy_name="top-k",
        strategy_instance=get_strategy("top-k"),
        k=2,
        k_agg=4,
        eval_concurrency=1,
        rounds=rounds,
        current_prompt="test prompt",
        use_reflection=use_reflection,
    )
    defaults.update(overrides)
    return WorkflowContext(**defaults)


# ── _build_entries ───────────────────────────────────────


class TestBuildEntries:
    def test_includes_model_and_score(self):
        c = _make_candidate(0.75, model="gpt-4o", output="x = 1")
        text = _build_entries([c])
        assert "gpt-4o" in text
        assert "0.750" in text

    def test_includes_stderr_when_present(self):
        c = _make_candidate(0.0, stderr="NameError: name 'x' is not defined", exit_code=1)
        text = _build_entries([c])
        assert "NameError" in text

    def test_no_stderr_section_when_clean(self):
        c = _make_candidate(1.0, stderr="", exit_code=0)
        text = _build_entries([c])
        assert "stderr" not in text

    def test_output_truncated(self):
        long_output = "x" * 2000
        c = _make_candidate(0.5, output=long_output)
        text = _build_entries([c], max_output_chars=100)
        # Should be cut short
        assert len(text) < 500


# ── reflect() ────────────────────────────────────────────


class TestReflect:
    def test_returns_none_when_no_candidates(self):
        result = reflect([], round_num=0, api_key="fake-key")
        assert result is None

    def test_returns_none_when_no_api_key(self):
        c = _make_candidate(0.5)
        result = reflect([c], round_num=0, api_key="")
        assert result is None

    def test_returns_brief_on_success(self):
        c = _make_candidate(0.3, stderr="TypeError: unsupported operand", exit_code=1)
        fake_response = {
            "choices": [{"message": {"content": "Diagnose: type error. Fix: cast to int first."}}]
        }
        with patch("fanout.reflect.httpx.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: fake_response,
                raise_for_status=lambda: None,
            )
            result = reflect([c], round_num=1, api_key="sk-test")

        assert result == "Diagnose: type error. Fix: cast to int first."

    def test_returns_none_on_http_error(self):
        import httpx
        c = _make_candidate(0.5)
        with patch("fanout.reflect.httpx.post", side_effect=httpx.ConnectError("refused")):
            result = reflect([c], round_num=0, api_key="sk-test")
        assert result is None

    def test_returns_none_on_empty_choices(self):
        c = _make_candidate(0.5)
        with patch("fanout.reflect.httpx.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"choices": []},
                raise_for_status=lambda: None,
            )
            result = reflect([c], round_num=0, api_key="sk-test")
        assert result is None


# ── reflect_step() ────────────────────────────────────────


class TestReflectStep:
    def test_no_op_when_use_reflection_false(self):
        ctx = _make_ctx(use_reflection=False)
        ctx.selected = [_make_candidate(0.5)]
        ctx.round_num = 0
        # Should not touch ctx.reflection at all
        reflect_step(ctx)
        assert ctx.reflection == ""

    def test_no_op_on_last_round(self):
        ctx = _make_ctx(rounds=2)
        ctx.selected = [_make_candidate(0.3, stderr="error")]
        ctx.round_num = 1  # last round (0-indexed, rounds=2)
        reflect_step(ctx)
        assert ctx.reflection == ""

    def test_no_op_when_no_selected(self):
        ctx = _make_ctx()
        ctx.selected = []
        ctx.round_num = 0
        reflect_step(ctx)
        assert ctx.reflection == ""

    def test_stores_brief_on_ctx(self):
        ctx = _make_ctx(rounds=3)
        ctx.selected = [_make_candidate(0.3, stderr="AssertionError")]
        ctx.round_num = 0

        with patch("fanout.reflect.reflect") as mock_reflect:
            mock_reflect.return_value = "Fix: handle edge case in loop."
            reflect_step(ctx)

        assert ctx.reflection == "Fix: handle edge case in loop."

    def test_graceful_on_none_brief(self):
        """If reflect() returns None (e.g. no API key), step is silent."""
        ctx = _make_ctx(rounds=3)
        ctx.selected = [_make_candidate(0.5)]
        ctx.round_num = 0

        with patch("fanout.reflect.reflect", return_value=None):
            reflect_step(ctx)  # must not raise

        assert ctx.reflection == ""

    def test_prints_brief_preview_to_console(self):
        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True, width=200)
        ctx = _make_ctx(rounds=3, console=test_console)
        ctx.selected = [_make_candidate(0.3)]
        ctx.round_num = 0

        with patch("fanout.reflect.reflect", return_value="Use binary search instead."):
            reflect_step(ctx)

        assert "reflection" in buf.getvalue()
        assert "binary search" in buf.getvalue()

    def test_reflection_consumed_by_evolve_step(self):
        """evolve_step prepends the brief and then clears ctx.reflection."""
        from fanout.workflow import evolve_step

        store = _make_store()
        ctx = _make_ctx(store=store, rounds=3)
        ctx.reflection = "Fix: use a hash map."
        ctx.round_num = 0
        sol = Solution(run_id=ctx.run.id, model="m", output="x")
        ev = Evaluation(solution_id=sol.id, evaluator="script", score=0.5)
        store.save_solution(sol)
        store.save_evaluation(ev)
        ctx.selected = [SolutionWithScores(solution=sol, evaluations=[ev], aggregate_score=0.5)]

        evolve_step(ctx)

        assert "Fix: use a hash map." in ctx.current_prompt
        assert "Improvement Brief" in ctx.current_prompt
        assert ctx.reflection == ""  # consumed
