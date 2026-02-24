"""smolagents Tool subclasses for the launch primitive."""

from __future__ import annotations

import asyncio
from typing import Any

from smolagents import Tool

from fanout.db.models import Solution
from fanout.solution_format import extract_solution
from fanout.store import Store


class ReadSolutionsTool(Tool):
    """Read all solutions for the current run, ranked by score."""

    name = "read_solutions"
    description = (
        "Read all solutions submitted so far for this run, ranked by aggregate score. "
        "Returns a list of solutions with their IDs, models, scores, and output previews."
    )
    inputs = {}
    output_type = "string"

    def __init__(self, store: Store, run_id: str, **kwargs: Any):
        super().__init__(**kwargs)
        self._store = store
        self._run_id = run_id

    def forward(self) -> str:
        scored = self._store.get_solutions_with_scores(self._run_id)
        if not scored:
            return "No solutions submitted yet."

        lines: list[str] = []
        for i, s in enumerate(scored[:10]):
            extracted = extract_solution(s.solution.output)
            preview = extracted[:2000]
            if len(extracted) > 2000:
                preview += "\n... (truncated)"
            lines.append(
                f"--- Solution {i+1} ---\n"
                f"ID: {s.solution.id}\n"
                f"Model: {s.solution.model}\n"
                f"Score: {s.aggregate_score:.4f}\n"
                f"Output:\n{preview}\n"
            )
        return "\n".join(lines)


class WriteSolutionTool(Tool):
    """Submit a new solution to the store."""

    name = "write_solution"
    description = (
        "Submit a new solution for the current run. "
        "Provide the full solution code/text as the 'solution' argument. "
        "Returns the solution ID."
    )
    inputs = {
        "solution": {
            "type": "string",
            "description": "The full solution code or text to submit.",
        },
    }
    output_type = "string"

    def __init__(self, store: Store, run_id: str, model: str, **kwargs: Any):
        super().__init__(**kwargs)
        self._store = store
        self._run_id = run_id
        self._model = model
        self._iteration = 0

    def forward(self, solution: str) -> str:
        self._iteration += 1
        sol = Solution(
            run_id=self._run_id,
            model=self._model,
            output=solution,
            metadata={"source": "agent", "iteration": self._iteration},
        )
        self._store.save_solution(sol)
        return f"Solution saved with ID: {sol.id} (iteration {self._iteration})"


class RunEvalTool(Tool):
    """Run the eval script against a solution."""

    name = "run_eval"
    description = (
        "Run the evaluation script against a solution by its ID. "
        "Returns the score (0-1) and any stderr output."
    )
    inputs = {
        "solution_id": {
            "type": "string",
            "description": "The ID of the solution to evaluate.",
        },
    }
    output_type = "string"

    def __init__(
        self,
        store: Store,
        eval_script: str,
        materializer: str = "file",
        file_ext: str = ".py",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._store = store
        self._eval_script = eval_script
        self._materializer = materializer
        self._file_ext = file_ext

    def forward(self, solution_id: str) -> str:
        sol = self._store.get_solution(solution_id)
        if sol is None:
            return f"Error: Solution {solution_id} not found."

        from fanout.evaluators.script import ScriptEvaluator

        evaluator = ScriptEvaluator()
        context = {
            "eval_script": self._eval_script,
            "materializer": self._materializer,
            "file_extension": self._file_ext,
        }

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(evaluator.evaluate(sol, context))
        finally:
            loop.close()

        # Save the evaluation
        from fanout.db.models import Evaluation

        ev = Evaluation(
            solution_id=solution_id,
            evaluator="script",
            score=result.score,
            raw_score=result.raw_score,
            details=result.details,
        )
        self._store.save_evaluation(ev)

        stderr = result.details.get("stderr", "")
        stdout = result.details.get("stdout", "")
        parts = [f"Score: {result.score:.4f}"]
        if stderr:
            parts.append(f"Stderr:\n{stderr}")
        if stdout:
            parts.append(f"Stdout:\n{stdout}")
        return "\n".join(parts)


class ReadPromptTool(Tool):
    """Read the original task prompt."""

    name = "read_prompt"
    description = "Read the original task prompt for this run."
    inputs = {}
    output_type = "string"

    def __init__(self, prompt: str, **kwargs: Any):
        super().__init__(**kwargs)
        self._prompt = prompt

    def forward(self) -> str:
        return self._prompt
