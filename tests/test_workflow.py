"""Tests for fanout.workflow — steps in isolation + full workflow."""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from fanout.channels.memory import MemoryChannel
from fanout.db.models import Evaluation, Run, Solution, SolutionWithScores
from fanout.providers.openrouter import SamplingConfig
from fanout.store import Store
from fanout.strategies.base import get_strategy
from fanout.workflow import (
    LaunchWorkflow,
    SampleWorkflow,
    Workflow,
    WorkflowContext,
    WorkflowResult,
    evaluate_step,
    evolve_step,
    launch_step,
    sample_step,
    select_step,
)


# ── Helpers ──────────────────────────────────────────────


def _make_store() -> Store:
    return Store(channel=MemoryChannel())


def _make_ctx(store: Store | None = None, **overrides) -> WorkflowContext:
    store = store or _make_store()
    run = Run(prompt="test prompt", total_rounds=3)
    store.save_run(run)
    defaults = dict(
        prompt="test prompt",
        store=store,
        run=run,
        config=SamplingConfig(models=["test-model"], n_samples=2),
        eval_context={"eval_script": "/bin/true", "materializer": "file"},
        evaluator_names=["script"],
        strategy_name="top-k",
        strategy_instance=get_strategy("top-k"),
        k=2,
        k_agg=4,
        eval_concurrency=1,
        rounds=3,
        current_prompt="test prompt",
    )
    defaults.update(overrides)
    return WorkflowContext(**defaults)


def _make_solution(run_id: str, round_num: int = 0, score: float = 0.5) -> tuple[Solution, Evaluation]:
    sol = Solution(run_id=run_id, round_num=round_num, model="test-model", output="print('hello')")
    ev = Evaluation(solution_id=sol.id, evaluator="script", score=score, raw_score=score)
    return sol, ev


# ── Step tests ───────────────────────────────────────────


class TestSampleStep:
    @patch("fanout.workflow.do_sample")
    def test_sets_solutions_on_ctx(self, mock_sample):
        ctx = _make_ctx()
        fake_solutions = [Solution(run_id=ctx.run.id, model="m", output="x")]
        mock_sample.return_value = fake_solutions

        sample_step(ctx)

        assert ctx.solutions is fake_solutions
        mock_sample.assert_called_once_with(
            ctx.current_prompt, ctx.config, ctx.store,
            ctx.run.id, ctx.round_num, ctx.parent_ids, api_key=ctx.api_key,
        )


class TestEvaluateStep:
    @patch("fanout.workflow.evaluate_solutions")
    def test_sets_evaluations_on_ctx(self, mock_eval):
        ctx = _make_ctx()
        sol = Solution(run_id=ctx.run.id, model="m", output="x")
        ctx.solutions = [sol]
        fake_evals = [Evaluation(solution_id=sol.id, evaluator="script", score=0.8)]
        mock_eval.return_value = fake_evals

        evaluate_step(ctx)

        assert ctx.evaluations is fake_evals
        mock_eval.assert_called_once_with(
            ctx.solutions, ctx.evaluator_names, ctx.store,
            ctx.eval_context, concurrency=ctx.eval_concurrency,
        )


class TestSelectStep:
    def test_updates_scores(self):
        store = _make_store()
        ctx = _make_ctx(store=store)

        # Seed store with solutions + evaluations
        sol, ev = _make_solution(ctx.run.id, round_num=0, score=0.75)
        store.save_solution(sol)
        store.save_evaluation(ev)

        select_step(ctx)

        assert len(ctx.selected) == 1
        assert ctx.selected[0].aggregate_score == 0.75
        assert ctx.round_scores == [0.75]
        assert ctx.best_score == 0.75

    def test_accumulates_best_score(self):
        store = _make_store()
        ctx = _make_ctx(store=store)

        # Round 0 — score 0.5
        sol0, ev0 = _make_solution(ctx.run.id, round_num=0, score=0.5)
        store.save_solution(sol0)
        store.save_evaluation(ev0)
        ctx.round_num = 0
        select_step(ctx)

        # Round 1 — score 0.9
        sol1, ev1 = _make_solution(ctx.run.id, round_num=1, score=0.9)
        store.save_solution(sol1)
        store.save_evaluation(ev1)
        ctx.round_num = 1
        select_step(ctx)

        assert ctx.best_score == 0.9
        assert ctx.round_scores == [0.5, 0.9]


class TestEvolveStep:
    def test_updates_run_round_and_parent_ids(self):
        store = _make_store()
        ctx = _make_ctx(store=store, rounds=3)
        sol, ev = _make_solution(ctx.run.id, round_num=1, score=0.8)
        store.save_solution(sol)
        store.save_evaluation(ev)
        ctx.selected = [SolutionWithScores(solution=sol, evaluations=[ev], aggregate_score=0.8)]
        ctx.round_num = 1  # round 1 so build_prompts actually rebuilds

        evolve_step(ctx)

        assert ctx.parent_ids == [sol.id]
        updated_run = store.get_run(ctx.run.id)
        assert updated_run.current_round == 2
        # current_prompt should have been rebuilt (not round 0, so strategy builds new prompt)
        assert ctx.current_prompt != "test prompt"

    def test_last_round_keeps_prompt(self):
        store = _make_store()
        ctx = _make_ctx(store=store, rounds=2)
        sol, ev = _make_solution(ctx.run.id, round_num=1, score=0.8)
        store.save_solution(sol)
        store.save_evaluation(ev)
        ctx.selected = [SolutionWithScores(solution=sol, evaluations=[ev], aggregate_score=0.8)]
        ctx.round_num = 1  # last round (rounds=2)

        evolve_step(ctx)

        # On last round, prompt should not be rebuilt
        assert ctx.current_prompt == "test prompt"


# ── SampleWorkflow ───────────────────────────────────────


class TestSampleWorkflow:
    @patch("fanout.workflow.do_sample")
    @patch("fanout.workflow.evaluate_solutions")
    def test_full_workflow_with_mocked_primitives(self, mock_eval, mock_sample):
        store = _make_store()

        def fake_sample(prompt, config, store, run_id, round_num, parent_ids, api_key=None):
            sol = Solution(run_id=run_id, round_num=round_num, model="m", output="code")
            store.save_solution(sol)
            return [sol]

        def fake_eval(solutions, evaluator_names, store, context, concurrency=1):
            evals = []
            for sol in solutions:
                ev = Evaluation(solution_id=sol.id, evaluator="script", score=0.6 + sol.round_num * 0.1)
                store.save_evaluation(ev)
                evals.append(ev)
            return evals

        mock_sample.side_effect = fake_sample
        mock_eval.side_effect = fake_eval

        wf = SampleWorkflow()
        result = wf.run(
            prompt="solve this",
            models=["test-model"],
            rounds=3,
            strategy="top-k",
            k=1,
            store=store,
        )

        assert isinstance(result, WorkflowResult)
        assert len(result.round_scores) == 3
        assert result.round_scores[0] == pytest.approx(0.6)
        assert result.round_scores[1] == pytest.approx(0.7)
        assert result.round_scores[2] == pytest.approx(0.8)
        assert result.best_score == pytest.approx(0.8)
        assert result.run_id

    @patch("fanout.workflow.do_sample")
    @patch("fanout.workflow.evaluate_solutions")
    def test_early_stop_via_extra_steps(self, mock_eval, mock_sample):
        store = _make_store()

        def fake_sample(prompt, config, store, run_id, round_num, parent_ids, api_key=None):
            sol = Solution(run_id=run_id, round_num=round_num, model="m", output="code")
            store.save_solution(sol)
            return [sol]

        def fake_eval(solutions, evaluator_names, store, context, concurrency=1):
            evals = []
            for sol in solutions:
                # Score 1.0 on round 1 to trigger early stop
                score = 1.0 if sol.round_num == 1 else 0.5
                ev = Evaluation(solution_id=sol.id, evaluator="script", score=score)
                store.save_evaluation(ev)
                evals.append(ev)
            return evals

        mock_sample.side_effect = fake_sample
        mock_eval.side_effect = fake_eval

        def stop_if_solved(ctx):
            if ctx.best_score >= 1.0:
                ctx.stop = True

        wf = SampleWorkflow(extra_steps=[stop_if_solved])
        result = wf.run(
            prompt="prove this",
            models=["test-model"],
            rounds=5,
            strategy="top-k",
            k=1,
            store=store,
        )

        assert result.best_score == pytest.approx(1.0)
        assert len(result.round_scores) == 2  # stopped after round 1

    @patch("fanout.workflow.do_sample")
    @patch("fanout.workflow.evaluate_solutions")
    def test_verbose_logging_runs_without_error(self, mock_eval, mock_sample):
        store = _make_store()

        def fake_sample(prompt, config, store, run_id, round_num, parent_ids, api_key=None):
            sol = Solution(run_id=run_id, round_num=round_num, model="m", output="print('hi')")
            store.save_solution(sol)
            return [sol]

        def fake_eval(solutions, evaluator_names, store, context, concurrency=1):
            evals = []
            for sol in solutions:
                ev = Evaluation(
                    solution_id=sol.id, evaluator="script",
                    score=0.5, exit_code=0, stderr="", stdout="ok",
                )
                store.save_evaluation(ev)
                evals.append(ev)
            return evals

        mock_sample.side_effect = fake_sample
        mock_eval.side_effect = fake_eval

        buf = StringIO()
        test_console = Console(file=buf, force_terminal=False)

        wf = SampleWorkflow()
        result = wf.run(
            prompt="solve this",
            models=["test-model"],
            rounds=2,
            strategy="top-k",
            k=1,
            store=store,
            verbose=True,
            console=test_console,
        )

        output = buf.getvalue()
        assert "Round 1/2" in output
        assert "Round 2/2" in output
        assert "sampled" in output
        assert "top=" in output
        assert "best=" in output
        assert result.best_score > 0


# ── LaunchWorkflow ───────────────────────────────────────


class TestLaunchWorkflow:
    @patch("fanout.workflow.do_launch")
    def test_launch_step_sets_solutions(self, mock_launch):
        ctx = _make_ctx()
        fake_solutions = [Solution(run_id=ctx.run.id, model="m", output="x")]
        mock_launch.return_value = fake_solutions

        launch_step(ctx, n_agents=3, max_steps=10)

        assert ctx.solutions is fake_solutions
        mock_launch.assert_called_once_with(
            prompt=ctx.prompt,
            models=ctx.config.models,
            store=ctx.store,
            run_id=ctx.run.id,
            n_agents=3,
            max_steps=10,
            eval_script=ctx.eval_context.get("eval_script"),
            materializer=ctx.eval_context.get("materializer", "file"),
            file_ext=ctx.eval_context.get("file_extension", ".py"),
            verbose=ctx.verbose,
            api_key=ctx.api_key,
        )

    @patch("fanout.workflow.do_launch")
    def test_launch_workflow_runs(self, mock_launch):
        store = _make_store()

        def fake_launch(**kwargs):
            sol = Solution(run_id=kwargs["run_id"], round_num=0, model="m", output="code")
            store.save_solution(sol)
            ev = Evaluation(solution_id=sol.id, evaluator="script", score=0.9, raw_score=0.9)
            store.save_evaluation(ev)
            return [sol]

        mock_launch.side_effect = fake_launch

        wf = LaunchWorkflow()
        result = wf.run(
            prompt="solve this",
            models=["test-model"],
            strategy="top-k",
            k=1,
            store=store,
            n_agents=3,
            max_steps=10,
        )

        assert isinstance(result, WorkflowResult)
        assert len(result.round_scores) == 1
        assert result.best_score == pytest.approx(0.9)
        mock_launch.assert_called_once()
