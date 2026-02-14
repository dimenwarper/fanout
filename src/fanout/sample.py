"""Sampling orchestration — fan out prompt to models and persist results."""

from __future__ import annotations

import asyncio

from fanout.db.models import Solution
from fanout.providers.openrouter import OpenRouterClient, SamplingConfig
from fanout.store import Store


async def sample_async(
    prompt: str,
    config: SamplingConfig,
    store: Store,
    run_id: str,
    round_num: int = 0,
    parent_solution_ids: list[str] | None = None,
    api_key: str | None = None,
) -> list[Solution]:
    """Sample from all configured models and save solutions."""
    client = OpenRouterClient(api_key=api_key)
    solutions = await client.sample(
        prompt=prompt,
        config=config,
        run_id=run_id,
        round_num=round_num,
        parent_solution_ids=parent_solution_ids,
    )
    for sol in solutions:
        store.save_solution(sol)
    return solutions


def sample(
    prompt: str,
    config: SamplingConfig,
    store: Store,
    run_id: str,
    round_num: int = 0,
    parent_solution_ids: list[str] | None = None,
    api_key: str | None = None,
) -> list[Solution]:
    """Synchronous wrapper around sample_async."""
    return asyncio.run(sample_async(
        prompt, config, store, run_id, round_num, parent_solution_ids, api_key,
    ))
