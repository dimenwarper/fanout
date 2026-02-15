"""Stdin materializer — signals that eval script should receive output via stdin."""

from __future__ import annotations

from pathlib import Path

from fanout.db.models import Solution
from fanout.materializers.base import BaseMaterializer, register_materializer

# Sentinel path returned to signal stdin piping to the script evaluator.
STDIN_SENTINEL = Path("__stdin__")


@register_materializer
class StdinMaterializer(BaseMaterializer):
    name = "stdin"
    description = "Pipes solution output to eval script via stdin"

    async def materialize(self, solution: Solution, workspace: Path, context: dict) -> Path:
        # Write a reference copy for debugging, but return the sentinel.
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "input.txt").write_text(solution.output)
        return STDIN_SENTINEL
