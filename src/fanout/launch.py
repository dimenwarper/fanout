"""Launch primitive — spawn concurrent smolagents that iteratively produce solutions."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import re

from rich.console import Console
from rich.console import Group
from rich.table import Table
from rich.text import Text
from rich.live import Live

from fanout.db.models import Solution
from fanout.store import Store

AGENT_SYSTEM_PROMPT = """\
You are a coding agent. Your goal is to produce the best possible solution to the task.

Follow this loop:
1. Read the task prompt using the `read_prompt` tool.
2. Read existing solutions using `read_solutions` to see what others have produced.
3. Write your solution using `write_solution`.
4. If an eval script is available, evaluate your solution with `run_eval`.
5. Read solutions again to see if others have submitted better solutions.
6. If you can improve, write an improved solution and evaluate it.
7. Repeat until you are confident your solution scores well or you run out of steps.

Focus on producing correct, high-quality solutions. Learn from other agents' solutions and scores.
"""

AGENT_SYSTEM_PROMPT_WITH_MEMORY = """\
You are a coding agent. Your goal is to produce the best possible solution to the task.
You have access to a shared memory bank that all agents can read and write.

Use the memory bank to:
- Record OBSERVATIONS about the task structure, constraints, or edge cases.
- Write HYPOTHESES before trying a new approach — what you think will work and why.
- Record LEARNINGS after evaluating a solution — what the score was, what worked or failed, and why.
- Share STRATEGIES so other agents can build on your high-level approaches.

Follow this loop:
1. Read the task prompt using `read_prompt`.
2. Read shared memories with `read_memories` — learn from what others have already discovered.
3. Read existing solutions with `read_solutions` — see what has been tried and scored.
4. If you have an insight or plan, record it as a memory with `write_memory` before coding.
5. Write your solution using `write_solution`.
6. If an eval script is available, evaluate your solution with `run_eval`.
7. Record a LEARNING memory with `write_memory` — note the score and what you learned.
8. Go back to step 2 and iterate: read memories, improve your approach, repeat.

The memory bank is your shared scratchpad. Use it proactively — both to share your insights
and to avoid repeating mistakes others have already made.
"""


_MEMORY_TYPE_STYLE: dict[str, str] = {
    "observation": "cyan",
    "hypothesis":  "yellow",
    "learning":    "green",
    "strategy":    "magenta",
}


class _AgentTracker:
    """Thread-safe tracker for agent step progress, rendered via Rich Live."""

    def __init__(self, agent_labels: list[str], max_steps: int, console: Console | None = None,
                 store: Store | None = None, run_id: str | None = None):
        self._lock = threading.Lock()
        self._max_steps = max_steps
        self._top_score: float = 0.0
        self._agents: dict[str, dict[str, Any]] = {
            label: {"step": 0, "tool": "", "obs": "", "status": "running"}
            for label in agent_labels
        }
        self._store = store
        self._run_id = run_id
        self._console = console or Console()
        self._live = Live(self._build_display(), console=self._console, refresh_per_second=4)

    _SCORE_RE = re.compile(r"Score:\s*([\d.]+)")

    def _build_memory_table(self) -> Table | None:
        """Build a compact memory table from the store (called inside lock)."""
        if not self._store or not self._run_id:
            return None
        memories = self._store.get_memories_for_run(self._run_id)
        if not memories:
            return None

        table = Table(show_header=True, header_style="dim", expand=False, pad_edge=False,
                      title="Memory Bank", title_style="dim blue")
        table.add_column("Type", style="bold", width=12)
        table.add_column("Agent", style="dim", width=18)
        table.add_column("Content", max_width=60)

        for mem in memories:
            style = _MEMORY_TYPE_STYLE.get(mem.memory_type, "white")
            content = mem.content if len(mem.content) <= 57 else mem.content[:54] + "..."
            agent = mem.agent_id if len(mem.agent_id) <= 18 else mem.agent_id[:15] + "..."
            table.add_row(f"[{style}]{mem.memory_type}[/]", agent, content)

        return table

    def _build_display(self) -> Group:
        with self._lock:
            score_line = Text(f"  Top score: {self._top_score:.4f}", style="bold green")

        table = self._build_table()
        mem_table = self._build_memory_table()
        if mem_table:
            return Group(score_line, table, Text(""), mem_table)
        return Group(score_line, table)

    def _build_table(self) -> Table:
        table = Table(show_header=True, header_style="dim", expand=False, pad_edge=False)
        table.add_column("Agent", style="cyan", min_width=20)
        table.add_column("Step", justify="right", min_width=8)
        table.add_column("Tool", min_width=16)
        table.add_column("Observation", max_width=60)

        with self._lock:
            for label, info in self._agents.items():
                status = info["status"]
                if status == "done":
                    style = "dim"
                    step_str = "done"
                elif status == "error":
                    style = "red"
                    step_str = "error"
                else:
                    style = ""
                    step_str = f"{info['step']}/{self._max_steps}"

                obs = info["obs"]
                if len(obs) > 57:
                    obs = obs[:57] + "..."

                table.add_row(label, step_str, info["tool"], obs, style=style)

        return table

    def update(self, label: str, step: int, tool: str, obs: str) -> None:
        with self._lock:
            self._agents[label]["step"] = step
            self._agents[label]["tool"] = tool
            self._agents[label]["obs"] = obs
            # Parse score from observation (e.g. "Score: 0.4932")
            m = self._SCORE_RE.search(obs)
            if m:
                score = float(m.group(1))
                if score > self._top_score:
                    self._top_score = score
        self._live.update(self._build_display())

    def mark_done(self, label: str) -> None:
        with self._lock:
            self._agents[label]["status"] = "done"
        self._live.update(self._build_display())

    def mark_error(self, label: str, err: str) -> None:
        with self._lock:
            self._agents[label]["status"] = "error"
            self._agents[label]["obs"] = err[:57]
        self._live.update(self._build_display())

    def __enter__(self):
        self._live.__enter__()
        return self

    def __exit__(self, *args):
        self._live.__exit__(*args)


def _make_step_callback(label: str, tracker: _AgentTracker):
    """Return a step_callbacks dict for a single agent."""
    from smolagents.memory import ActionStep

    def on_step(step: ActionStep):
        tool = ""
        obs = ""
        if step.tool_calls:
            tool = step.tool_calls[0].name
        if step.observations:
            # Take first line, strip whitespace
            obs = step.observations.strip().splitlines()[0]
        elif step.error:
            obs = str(step.error).strip().splitlines()[0]
        tracker.update(label, step.step_number, tool, obs)

    return {ActionStep: on_step}


def _run_single_agent(
    prompt: str,
    model: str,
    store: Store,
    run_id: str,
    max_steps: int,
    eval_script: str | None,
    materializer: str,
    file_ext: str,
    api_key: str | None,
    label: str,
    tracker: _AgentTracker | None,
    use_memory: bool = False,
) -> list[Solution]:
    """Run a single smolagents ToolCallingAgent. Returns solutions it produced."""
    from smolagents import ToolCallingAgent
    from smolagents.monitoring import LogLevel

    from fanout.agent_tools import (
        ReadMemoriesTool,
        ReadPromptTool,
        ReadSolutionsTool,
        RunEvalTool,
        WriteSolutionTool,
        WriteMemoryTool,
    )

    # Derive a short, stable agent ID from the label for memory authorship
    agent_id = label.split("(")[0].strip().lower().replace(" ", "-")

    tools: list[Any] = [
        ReadSolutionsTool(store=store, run_id=run_id),
        WriteSolutionTool(store=store, run_id=run_id, model=model),
        ReadPromptTool(prompt=prompt),
    ]

    if use_memory:
        tools += [
            WriteMemoryTool(store=store, run_id=run_id, agent_id=agent_id),
            ReadMemoriesTool(store=store, run_id=run_id, synthesize_model=model),
        ]

    if eval_script:
        tools.append(
            RunEvalTool(
                store=store,
                eval_script=eval_script,
                materializer=materializer,
                file_ext=file_ext,
            )
        )

    llm = _make_model(model, api_key)

    step_callbacks = _make_step_callback(label, tracker) if tracker else None

    system_prompt = AGENT_SYSTEM_PROMPT_WITH_MEMORY if use_memory else AGENT_SYSTEM_PROMPT

    agent = ToolCallingAgent(
        tools=tools,
        model=llm,
        max_steps=max_steps,
        instructions=system_prompt,
        verbosity_level=LogLevel.OFF,
        step_callbacks=step_callbacks,
    )

    task_msg = (
        "Read the task prompt, then iteratively produce and improve solutions. "
        "Use the available tools to read the prompt, submit solutions, evaluate them, "
        "and read what other agents have produced."
    )
    agent.run(task_msg)

    # Collect all solutions this agent produced (by model name + source metadata)
    all_solutions = store.get_solutions_for_run(run_id)
    return [
        s for s in all_solutions
        if s.model == model and s.metadata.get("source") == "agent"
    ]


def _make_model(model: str, api_key: str | None) -> Any:
    """Create a smolagents model instance for the given model string."""
    from smolagents import OpenAIServerModel

    resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    return OpenAIServerModel(
        model_id=model,
        api_base="https://openrouter.ai/api/v1",
        api_key=resolved_key,
    )


def launch(
    prompt: str,
    models: list[str],
    store: Store,
    run_id: str,
    n_agents: int = 3,
    max_steps: int = 10,
    eval_script: str | None = None,
    materializer: str = "file",
    file_ext: str = ".py",
    concurrency: int | None = None,
    verbose: bool = False,
    api_key: str | None = None,
    console: Console | None = None,
    use_memory: bool = False,
) -> list[Solution]:
    """Launch concurrent agents that iteratively produce solutions.

    Models are distributed round-robin across agents.
    Returns all solutions produced during the launch.

    When ``use_memory=True`` each agent is given ``write_memory`` and
    ``read_memories`` tools and a memory-aware system prompt, enabling
    them to share observations, hypotheses, and learnings via a shared
    memory bank backed by the same Store as solutions.
    """
    max_workers = concurrency or n_agents

    # Distribute models round-robin
    agent_models = [models[i % len(models)] for i in range(n_agents)]
    agent_labels = [f"Agent {i+1} ({m})" for i, m in enumerate(agent_models)]

    all_solutions: list[Solution] = []

    tracker = _AgentTracker(
        agent_labels, max_steps, console=console,
        store=store if use_memory else None, run_id=run_id if use_memory else None,
    ) if verbose else None

    ctx_manager = tracker if tracker else _nullcontext()
    with ctx_manager:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_single_agent,
                    prompt=prompt,
                    model=agent_model,
                    store=store,
                    run_id=run_id,
                    max_steps=max_steps,
                    eval_script=eval_script,
                    materializer=materializer,
                    file_ext=file_ext,
                    api_key=api_key,
                    label=label,
                    tracker=tracker,
                    use_memory=use_memory,
                ): (agent_model, label)
                for agent_model, label in zip(agent_models, agent_labels)
            }

            for future in as_completed(futures):
                model_name, label = futures[future]
                try:
                    solutions = future.result()
                    all_solutions.extend(solutions)
                    if tracker:
                        tracker.mark_done(label)
                except Exception as e:
                    if tracker:
                        tracker.mark_error(label, str(e))
                    elif verbose:
                        import sys
                        print(f"Agent ({model_name}) failed: {e}", file=sys.stderr)

    return all_solutions


class _nullcontext:
    def __enter__(self): return self
    def __exit__(self, *args): pass
