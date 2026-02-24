"""Launch primitive — spawn concurrent smolagents that iteratively produce solutions."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

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


def _run_single_agent(
    prompt: str,
    model: str,
    store: Store,
    run_id: str,
    max_steps: int,
    eval_script: str | None,
    materializer: str,
    file_ext: str,
    verbose: bool,
    api_key: str | None,
) -> list[Solution]:
    """Run a single smolagents ToolCallingAgent. Returns solutions it produced."""
    from smolagents import ToolCallingAgent

    from fanout.agent_tools import (
        ReadPromptTool,
        ReadSolutionsTool,
        RunEvalTool,
        WriteSolutionTool,
    )

    tools: list[Any] = [
        ReadSolutionsTool(store=store, run_id=run_id),
        WriteSolutionTool(store=store, run_id=run_id, model=model),
        ReadPromptTool(prompt=prompt),
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

    agent = ToolCallingAgent(
        tools=tools,
        model=llm,
        max_steps=max_steps,
        system_prompt=AGENT_SYSTEM_PROMPT,
    )

    task_msg = (
        "Read the task prompt, then iteratively produce and improve solutions. "
        "Use the available tools to read the prompt, submit solutions, evaluate them, "
        "and read what other agents have produced."
    )
    agent.run(task_msg, verbose=verbose)

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
) -> list[Solution]:
    """Launch concurrent agents that iteratively produce solutions.

    Models are distributed round-robin across agents.
    Returns all solutions produced during the launch.
    """
    max_workers = concurrency or n_agents

    # Distribute models round-robin
    agent_models = [models[i % len(models)] for i in range(n_agents)]

    all_solutions: list[Solution] = []

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
                verbose=verbose,
                api_key=api_key,
            ): agent_model
            for agent_model in agent_models
        }

        for future in as_completed(futures):
            model_name = futures[future]
            try:
                solutions = future.result()
                all_solutions.extend(solutions)
            except Exception as e:
                if verbose:
                    import sys
                    print(f"Agent ({model_name}) failed: {e}", file=sys.stderr)

    return all_solutions
