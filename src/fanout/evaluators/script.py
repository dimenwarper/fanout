"""Script evaluator — runs a user-provided eval script against materialized solutions."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fanout.db.models import Solution
from fanout.evaluators.base import BaseEvaluator, EvaluatorResult, register_evaluator
from fanout.materializers.base import get_materializer
from fanout.solution_format import extract_solution
from fanout.materializers.stdin import STDIN_SENTINEL

DEFAULT_EVAL_TIMEOUT = 60  # seconds


@register_evaluator
class ScriptEvaluator(BaseEvaluator):
    name = "script"
    description = "Runs a user-provided eval script against materialized solution output"

    async def evaluate(self, solution: Solution, context: dict[str, Any] | None = None) -> EvaluatorResult:
        ctx = context or {}
        eval_script = ctx.get("eval_script")
        if not eval_script:
            raise ValueError("ScriptEvaluator requires 'eval_script' in context")

        timeout = ctx.get("eval_timeout", DEFAULT_EVAL_TIMEOUT)
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
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(input=extract_solution(solution.output).encode()),
                    timeout=timeout,
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    eval_script, str(materialized_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )

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
        except asyncio.TimeoutError:
            # Kill the process if it's still running
            try:
                proc.kill()
                await proc.wait()
            except (ProcessLookupError, UnboundLocalError):
                pass
            return EvaluatorResult(
                score=0.0,
                raw_score=0.0,
                details={
                    "stdout": "",
                    "stderr": f"Evaluation timed out after {timeout}s",
                    "exit_code": -1,
                },
            )
        finally:
            await materializer.cleanup(workspace)
