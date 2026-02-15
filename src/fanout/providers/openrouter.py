"""OpenRouter async client for LLM sampling."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
from pydantic import BaseModel, Field

from fanout.db.models import Solution

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class SamplingConfig(BaseModel):
    """Configuration for a sampling request."""

    models: list[str] = Field(default_factory=lambda: ["openai/gpt-4o-mini"])
    temperature: float = 0.7
    max_tokens: int = 2048
    n_per_model: int = 1
    model_set: str | None = None
    n_samples: int = 5


class OpenRouterClient:
    """Async client for the OpenRouter API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set. Pass it directly or set the env var.")

    async def sample(
        self,
        prompt: str,
        config: SamplingConfig,
        run_id: str = "",
        round_num: int = 0,
        parent_solution_ids: list[str] | None = None,
    ) -> list[Solution]:
        """Fan out prompt to all configured models, return Solutions."""
        solutions: list[Solution] = []
        parents = parent_solution_ids or []

        if config.model_set:
            from fanout.model_sets import get_model_set, pick_models

            ms = get_model_set(config.model_set)
            drawn_models = pick_models(ms, config.n_samples)

            async with httpx.AsyncClient(timeout=120) as client:
                for i, model in enumerate(drawn_models):
                    parent_id = parents[i % len(parents)] if parents else None
                    sol = await self._call_model(
                        client, prompt, model, config, run_id, round_num, parent_id,
                    )
                    solutions.append(sol)
        else:
            async with httpx.AsyncClient(timeout=120) as client:
                for model in config.models:
                    for i in range(config.n_per_model):
                        parent_id = parents[i % len(parents)] if parents else None
                        sol = await self._call_model(
                            client, prompt, model, config, run_id, round_num, parent_id,
                        )
                        solutions.append(sol)

        return solutions

    async def _call_model(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        model: str,
        config: SamplingConfig,
        run_id: str,
        round_num: int,
        parent_solution_id: str | None,
    ) -> Solution:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        start = time.perf_counter()
        resp = await client.post(OPENROUTER_API_URL, json=payload, headers=headers)
        latency_ms = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        return Solution(
            run_id=run_id,
            round_num=round_num,
            model=model,
            output=choice["message"]["content"],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cost_usd=0.0,  # OpenRouter may provide this in headers
            parent_solution_id=parent_solution_id,
        )
