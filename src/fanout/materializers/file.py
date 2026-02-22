"""File materializer — writes solution output to a file in the workspace."""

from __future__ import annotations

import re
from pathlib import Path

from fanout.db.models import Solution
from fanout.materializers.base import BaseMaterializer, register_materializer


def extract_solution(text: str) -> str:
    """Extract the final solution from LLM output.

    Looks for <solution>...</solution> tags first. Falls back to stripping
    markdown code fences. Returns the raw text if neither is found.
    """
    # Try <solution> tags (last match wins, in case CoT contains earlier tags)
    matches = re.findall(r"<solution>(.*?)</solution>", text, re.DOTALL)
    if matches:
        return strip_code_fences(matches[-1].strip())

    # Fallback: strip code fences from the whole output
    return strip_code_fences(text)


def strip_code_fences(text: str) -> str:
    """Strip markdown code fences (```python ... ```) from LLM output."""
    stripped = text.strip()
    stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
    stripped = re.sub(r"\n?```$", "", stripped.strip())
    return stripped


@register_materializer
class FileMaterializer(BaseMaterializer):
    name = "file"
    description = "Writes solution output to a file (default: output.py)"

    async def materialize(self, solution: Solution, workspace: Path, context: dict) -> Path:
        ext = context.get("file_extension", ".py")
        out_path = workspace / f"output{ext}"
        workspace.mkdir(parents=True, exist_ok=True)
        out_path.write_text(extract_solution(solution.output))
        return out_path
