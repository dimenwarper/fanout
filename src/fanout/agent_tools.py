"""smolagents Tool subclasses for the launch primitive."""

from __future__ import annotations

import asyncio
from typing import Any

from smolagents import Tool

from fanout.db.models import Memory, Solution
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


class WriteMemoryTool(Tool):
    """Write an observation, hypothesis, learning, or strategy to the shared memory bank."""

    name = "write_memory"
    description = (
        "Write a memory to the shared memory bank so other agents can learn from it. "
        "Use 'observation' for task insights, 'hypothesis' for ideas you plan to try, "
        "'learning' for outcomes after evaluating a solution (what worked or failed and why), "
        "and 'strategy' for high-level approaches worth sharing."
    )
    inputs = {
        "memory_type": {
            "type": "string",
            "description": (
                "Type of memory: 'observation' (task insight), 'hypothesis' (idea to try), "
                "'learning' (outcome from testing), or 'strategy' (high-level approach)."
            ),
        },
        "content": {
            "type": "string",
            "description": "The memory content. Be specific and actionable.",
        },
        "solution_id": {
            "type": "string",
            "description": "Optional ID of the solution this memory relates to.",
            "nullable": True,
        },
    }
    output_type = "string"

    _VALID_TYPES = {"observation", "hypothesis", "learning", "strategy"}

    def __init__(self, store: Store, run_id: str, agent_id: str, **kwargs: Any):
        super().__init__(**kwargs)
        self._store = store
        self._run_id = run_id
        self._agent_id = agent_id

    def forward(
        self,
        memory_type: str,
        content: str,
        solution_id: str | None = None,
    ) -> str:
        if memory_type not in self._VALID_TYPES:
            return f"Error: memory_type must be one of {sorted(self._VALID_TYPES)}"

        # Look up the solution's current aggregate score if an ID was provided
        score: float | None = None
        if solution_id:
            for s in self._store.get_solutions_with_scores(self._run_id):
                if s.solution.id == solution_id:
                    score = s.aggregate_score
                    break

        mem = Memory(
            run_id=self._run_id,
            agent_id=self._agent_id,
            memory_type=memory_type,
            content=content,
            solution_id=solution_id,
            score=score,
        )
        self._store.save_memory(mem)
        preview = content[:80] + ("..." if len(content) > 80 else "")
        return f"Memory saved [{memory_type}]: {preview}"


class ReadMemoriesTool(Tool):
    """Read shared memories from the memory bank."""

    name = "read_memories"
    description = (
        "Read shared memories from all agents — observations, hypotheses, learnings, "
        "and strategies. Call this before trying a new approach to build on what "
        "others have already discovered."
    )
    inputs = {
        "memory_type": {
            "type": "string",
            "description": (
                "Filter by type ('observation', 'hypothesis', 'learning', 'strategy'), "
                "or omit / pass 'all' to read everything."
            ),
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, store: Store, run_id: str, **kwargs: Any):
        super().__init__(**kwargs)
        self._store = store
        self._run_id = run_id

    def forward(self, memory_type: str | None = None) -> str:
        mtype = None if (not memory_type or memory_type == "all") else memory_type
        memories = self._store.get_memories_for_run(self._run_id, memory_type=mtype)

        if not memories:
            return "The memory bank is empty — you are the first to record learnings."

        lines = [f"=== Shared Memory Bank ({len(memories)} entr{'y' if len(memories)==1 else 'ies'}) ===\n"]
        for mem in memories:
            score_str = f"  score={mem.score:.3f}" if mem.score is not None else ""
            sol_str = f"  sol={mem.solution_id}" if mem.solution_id else ""
            lines.append(
                f"[{mem.memory_type.upper()}] {mem.agent_id}{score_str}{sol_str}\n"
                f"  {mem.content}\n"
            )
        return "\n".join(lines)


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
