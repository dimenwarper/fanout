"""Worktree materializer — creates a git worktree and applies a diff."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from fanout.db.models import Solution
from fanout.materializers.base import BaseMaterializer, register_materializer


async def _run(cmd: list[str], **kwargs) -> asyncio.subprocess.Process:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **kwargs,
    )
    await proc.wait()
    return proc


@register_materializer
class WorktreeMaterializer(BaseMaterializer):
    name = "worktree"
    description = "Creates a git worktree from HEAD and applies solution output as a unified diff"

    async def materialize(self, solution: Solution, workspace: Path, context: dict) -> Path:
        workspace.mkdir(parents=True, exist_ok=True)

        # Create a detached worktree at workspace
        proc = await _run(["git", "worktree", "add", "--detach", str(workspace)])
        if proc.returncode != 0:
            stderr = (await proc.stderr.read()).decode() if proc.stderr else ""
            raise RuntimeError(f"git worktree add failed: {stderr}")

        # Write the diff and apply it
        diff_path = workspace / "__fanout_patch.diff"
        diff_path.write_text(solution.output)

        proc = await _run(["git", "apply", str(diff_path)], cwd=workspace)
        if proc.returncode != 0:
            stderr = (await proc.stderr.read()).decode() if proc.stderr else ""
            raise RuntimeError(f"git apply failed: {stderr}")

        diff_path.unlink(missing_ok=True)
        return workspace

    async def cleanup(self, workspace: Path) -> None:
        proc = await _run(["git", "worktree", "remove", "--force", str(workspace)])
        if proc.returncode != 0:
            # Fallback: just remove the directory
            shutil.rmtree(workspace, ignore_errors=True)
