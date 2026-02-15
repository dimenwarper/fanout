"""Script evaluator — runs a user-provided eval script against materialized solutions."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fanout.db.models import Solution
from fanout.evaluators.base import BaseEvaluator, EvaluatorResult, register_evaluator
from fanout.materializers.base import get_materializer
from fanout.materializers.stdin import STDIN_SENTINEL


@register_evaluator
class ScriptEvaluator(BaseEvaluator):
    name = "script"
    description = "Runs a user-provided eval script against materialized solution output"

    async def evaluate(self, solution: Solution, context: dict[str, Any] | None = None) -> EvaluatorResult:
        ctx = context or {}
        eval_script = ctx.get("eval_script")
        if not eval_script:
            raise ValueError("ScriptEvaluator requires 'eval_script' in context")

        materializer_name = ctx.get("materializer", "file")
        materializer = get_materializer(materializer_name)

        run_id = solution.run_id
        sol_id = solution.id
        workspace = Path(".fanout") / "workspace" / run_id / sol_id
        workspace.mkdir(parents=True, exist_ok=True)

        try:
            materialized_path = await materializer.materialize(solution, workspace, ctx)

            # Build subprocess command
            if materialized_path == STDIN_SENTINEL:
                proc = await asyncio.create_subprocess_exec(
                    eval_script,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout_bytes, stderr_bytes = await proc.communicate(input=solution.output.encode())
            else:
                proc = await asyncio.create_subprocess_exec(
                    eval_script, str(materialized_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout_bytes, stderr_bytes = await proc.communicate()

            stdout = stdout_bytes.decode().strip()
            stderr = stderr_bytes.decode().strip()
            exit_code = proc.returncode

            # Parse score from last line of stdout
            score = 0.0
            if stdout:
                last_line = stdout.splitlines()[-1].strip()
                try:
                    score = float(last_line)
                except ValueError:
                    pass

            return EvaluatorResult(
                score=score,
                raw_score=score,
                details={"stdout": stdout, "stderr": stderr, "exit_code": exit_code},
            )
        finally:
            await materializer.cleanup(workspace)
