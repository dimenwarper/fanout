"""Pydantic data models for Fanout."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class Run(BaseModel):
    """A sampling run — one prompt through N rounds of fan-out."""

    id: str = Field(default_factory=_new_id)
    prompt: str
    config: dict[str, Any] = Field(default_factory=dict)
    current_round: int = 0
    total_rounds: int = 1
    parent_run_id: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Solution(BaseModel):
    """A single model response."""

    id: str = Field(default_factory=_new_id)
    run_id: str
    round_num: int = 0
    model: str
    output: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    parent_solution_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Evaluation(BaseModel):
    """A single evaluator score for a solution."""

    id: str = Field(default_factory=_new_id)
    solution_id: str
    evaluator: str
    score: float  # normalized 0-1
    raw_score: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SolutionWithScores(BaseModel):
    """Read-side composite: solution + all evaluation scores."""

    solution: Solution
    evaluations: list[Evaluation] = Field(default_factory=list)
    aggregate_score: float = 0.0

    @property
    def scores_by_evaluator(self) -> dict[str, float]:
        return {e.evaluator: e.score for e in self.evaluations}
