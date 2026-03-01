"""Post-run reporting: LLM summary generation and record saving."""

from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import httpx

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fanout.solution_format import extract_solution
from fanout.store import Store

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

_MEMORY_TYPE_STYLE: dict[str, str] = {
    "observation": "cyan",
    "hypothesis": "yellow",
    "learning": "green",
    "strategy": "magenta",
}


def print_memory_summary(
    results: list[dict[str, Any]],
    store: Store,
    console: Console,
) -> None:
    """Print a memory bank panel for each task that has memories."""
    for r in results:
        memories = store.get_memories_for_run(r["run_id"])
        if not memories:
            continue
        mem_table = Table(show_header=True, header_style="bold", expand=True, box=None)
        mem_table.add_column("Type", style="bold", width=12)
        mem_table.add_column("Agent", style="dim", width=22)
        mem_table.add_column("Score", justify="right", width=7)
        mem_table.add_column("Content")
        for mem in memories:
            style = _MEMORY_TYPE_STYLE.get(mem.memory_type, "white")
            score_str = f"{mem.score:.3f}" if mem.score is not None else "—"
            content = mem.content if len(mem.content) <= 120 else mem.content[:117] + "..."
            mem_table.add_row(
                f"[{style}]{mem.memory_type}[/]", mem.agent_id,
                score_str, content,
            )
        task = r.get("task", r["run_id"][:8])
        console.print(Panel(
            mem_table,
            title=f"[bold]Memory Bank — {task}[/] ({len(memories)} entries)",
            border_style="dim blue",
        ))


async def generate_summary(
    results: list[dict[str, Any]],
    store: Store,
    model: str = "anthropic/claude-sonnet-4-5",
    top_k: int = 3,
    api_key: str | None = None,
) -> str:
    """Call LLM to summarize what made top solutions successful."""
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return "(skipped summary — OPENROUTER_API_KEY not set)"

    # Build context from top solutions across all tasks
    sections: list[str] = []
    for r in results:
        run_id = r["run_id"]
        task = r.get("task", run_id)
        strategy = r.get("strategy", "unknown")
        scored = store.get_solutions_with_scores(run_id)
        top = scored[:top_k]
        if not top:
            continue

        lines = [f"[Task: {task}, Strategy: {strategy}]"]
        for i, sw in enumerate(top, 1):
            code = extract_solution(sw.solution.output)
            # Truncate very long solutions to keep prompt manageable
            if len(code) > 3000:
                code = code[:3000] + "\n... (truncated)"
            lines.append(
                f"Solution {i} (score={sw.aggregate_score:.4f}, model={sw.solution.model}):\n{code}"
            )
        sections.append("\n".join(lines))

    if not sections:
        return "(no solutions to summarize)"

    prompt = (
        "You are analyzing solutions from a code optimization benchmark.\n\n"
        "For each task below, I'll show the top solutions with their scores.\n"
        "Summarize:\n"
        "- What patterns or techniques appear in winning solutions\n"
        "- Key differences between high and low scoring solutions\n"
        "- Actionable insights for improving future runs\n\n"
        + "\n\n".join(sections)
    )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 2048,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(OPENROUTER_API_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def _build_scores_csv(results: list[dict[str, Any]], store: Store | None = None) -> str:
    """Build a CSV with step, task, strategy, max_score columns.

    For launch/agent mode, solutions are ordered by submission time so
    the CSV shows a proper progression instead of a single step.
    """
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(["step", "task", "strategy", "max_score"])
    for r in results:
        task = r.get("task", "task")
        strategy = r.get("strategy", "unknown")
        mode = r.get("mode", "sample")

        if mode == "agent" and store is not None:
            # Pull solutions sorted by submission time for proper progression
            scored = store.get_solutions_with_scores(r["run_id"])
            if scored:
                by_time = sorted(scored, key=lambda s: s.solution.created_at)
                running_max = 0.0
                for step, sw in enumerate(by_time, 1):
                    running_max = max(running_max, sw.aggregate_score)
                    writer.writerow([step, task, strategy, f"{running_max:.4f}"])
                continue

        # Sample mode: use round_scores as before
        running_max = 0.0
        for step, score in enumerate(r.get("round_scores", []), 1):
            running_max = max(running_max, score)
            writer.writerow([step, task, strategy, f"{running_max:.4f}"])
    return buf.getvalue()


def save_record(
    results: list[dict[str, Any]],
    store: Store,
    output_dir: Path,
    name: str | None = None,
    summary: str | None = None,
    top_k: int = 3,
    cli_args: dict[str, Any] | None = None,
) -> Path:
    """Save solutions and report to a run directory.

    Returns the path to the created directory.
    """
    # Use name if provided, otherwise first run_id
    run_id = results[0]["run_id"] if results else "unknown"
    dir_name = name or run_id[:8]
    run_dir = output_dir / dir_name

    # If directory already exists, ask for confirmation before overwriting
    if run_dir.exists():
        answer = input(f"  Record directory '{run_dir}' already exists. Overwrite? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("  Skipping record save.")
            return run_dir
        shutil.rmtree(run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)

    # manifest.json
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "tasks": [r.get("task") for r in results],
        "strategies": list({r.get("strategy") for r in results}),
        "run_ids": [r["run_id"] for r in results],
    }
    if cli_args:
        manifest["cli_args"] = cli_args
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # results.json
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))

    # scores.csv — step-by-step max score progression
    (run_dir / "scores.csv").write_text(_build_scores_csv(results, store=store))

    # solutions/
    sol_dir = run_dir / "solutions"
    sol_dir.mkdir(exist_ok=True)
    for r in results:
        run_id = r["run_id"]
        task = r.get("task", "task")
        scored = store.get_solutions_with_scores(run_id)
        for i, sw in enumerate(scored[:top_k], 1):
            code = extract_solution(sw.solution.output)
            ext = ".lean" if "lean" in sw.solution.output.lower() else ".py"
            (sol_dir / f"{task}_{i}{ext}").write_text(code)

    # summary.md
    if summary:
        (run_dir / "summary.md").write_text(summary)

    return run_dir
