"""Tests for the launch primitive — agent tools and orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fanout.channels.memory import MemoryChannel
from fanout.db.models import Run, Solution
from fanout.store import Store


def _make_store() -> Store:
    return Store(channel=MemoryChannel())


def _seed_run(store: Store, prompt: str = "Write hello world") -> str:
    run = Run(prompt=prompt)
    store.save_run(run)
    return run.id


# ── Tool unit tests ──────────────────────────────────────


class TestReadPromptTool:
    def test_returns_prompt(self):
        from fanout.agent_tools import ReadPromptTool

        tool = ReadPromptTool(prompt="Write a sort function")
        assert tool.forward() == "Write a sort function"


class TestWriteSolutionTool:
    def test_saves_solution(self):
        from fanout.agent_tools import WriteSolutionTool

        store = _make_store()
        run_id = _seed_run(store)
        tool = WriteSolutionTool(store=store, run_id=run_id, model="test-model")

        result = tool.forward(solution="def hello(): pass")
        assert "saved" in result.lower()

        solutions = store.get_solutions_for_run(run_id)
        assert len(solutions) == 1
        assert solutions[0].output == "def hello(): pass"
        assert solutions[0].model == "test-model"
        assert solutions[0].metadata["source"] == "agent"
        assert solutions[0].metadata["iteration"] == 1

    def test_tracks_iterations(self):
        from fanout.agent_tools import WriteSolutionTool

        store = _make_store()
        run_id = _seed_run(store)
        tool = WriteSolutionTool(store=store, run_id=run_id, model="test-model")

        tool.forward(solution="v1")
        tool.forward(solution="v2")

        solutions = store.get_solutions_for_run(run_id)
        iterations = sorted(s.metadata["iteration"] for s in solutions)
        assert iterations == [1, 2]


class TestReadSolutionsTool:
    def test_no_solutions(self):
        from fanout.agent_tools import ReadSolutionsTool

        store = _make_store()
        run_id = _seed_run(store)
        tool = ReadSolutionsTool(store=store, run_id=run_id)

        result = tool.forward()
        assert "no solutions" in result.lower()

    def test_shows_solutions(self):
        from fanout.agent_tools import ReadSolutionsTool

        store = _make_store()
        run_id = _seed_run(store)

        sol = Solution(run_id=run_id, model="m1", output="print('hi')")
        store.save_solution(sol)

        tool = ReadSolutionsTool(store=store, run_id=run_id)
        result = tool.forward()
        assert sol.id in result
        assert "m1" in result


class TestRunEvalTool:
    def test_solution_not_found(self):
        from fanout.agent_tools import RunEvalTool

        store = _make_store()
        tool = RunEvalTool(store=store, eval_script="./eval.sh")

        result = tool.forward(solution_id="nonexistent")
        assert "not found" in result.lower()


# ── Store.get_solution test ──────────────────────────────


class TestStoreGetSolution:
    def test_get_existing(self):
        store = _make_store()
        run_id = _seed_run(store)
        sol = Solution(run_id=run_id, model="m1", output="code")
        store.save_solution(sol)

        fetched = store.get_solution(sol.id)
        assert fetched is not None
        assert fetched.id == sol.id
        assert fetched.output == "code"

    def test_get_missing(self):
        store = _make_store()
        assert store.get_solution("nope") is None


# ── MemoryChannel thread safety ──────────────────────────


class TestMemoryChannelThreadSafety:
    def test_concurrent_puts(self):
        import threading

        ch = MemoryChannel()
        errors: list[Exception] = []

        def writer(n: int):
            try:
                for i in range(50):
                    ch.put("test", f"key-{n}-{i}", {"v": n * 100 + i})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        items = ch.list("test")
        assert len(items) == 200  # 4 threads × 50 keys


# ── Launch orchestration test ────────────────────────────


class TestLaunchOrchestration:
    @patch("fanout.launch._run_single_agent")
    def test_launch_distributes_models(self, mock_run):
        """launch() distributes models round-robin and collects results."""
        store = _make_store()
        run_id = _seed_run(store)

        sol1 = Solution(run_id=run_id, model="m1", output="s1", metadata={"source": "agent"})
        sol2 = Solution(run_id=run_id, model="m2", output="s2", metadata={"source": "agent"})
        sol3 = Solution(run_id=run_id, model="m1", output="s3", metadata={"source": "agent"})

        # Each call returns solutions for that agent
        mock_run.side_effect = [[sol1], [sol2], [sol3]]

        from fanout.launch import launch

        solutions = launch(
            prompt="test",
            models=["m1", "m2"],
            store=store,
            run_id=run_id,
            n_agents=3,
            max_steps=5,
        )

        assert len(solutions) == 3
        assert mock_run.call_count == 3

        # Check round-robin model distribution
        call_models = [call.kwargs["model"] for call in mock_run.call_args_list]
        assert call_models == ["m1", "m2", "m1"]

    @patch("fanout.launch._run_single_agent")
    def test_launch_handles_agent_failure(self, mock_run):
        """launch() continues even if one agent fails."""
        store = _make_store()
        run_id = _seed_run(store)

        sol = Solution(run_id=run_id, model="m1", output="ok", metadata={"source": "agent"})
        mock_run.side_effect = [[sol], RuntimeError("boom")]

        from fanout.launch import launch

        solutions = launch(
            prompt="test",
            models=["m1"],
            store=store,
            run_id=run_id,
            n_agents=2,
            max_steps=5,
        )

        assert len(solutions) == 1
        assert solutions[0].output == "ok"
