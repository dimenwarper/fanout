"""Latency evaluator — scores solutions by response time."""

from __future__ import annotations

from typing import Any

from fanout.db.models import Solution
from fanout.evaluators.base import BaseEvaluator, EvaluatorResult, register_evaluator


@register_evaluator
class LatencyEvaluator(BaseEvaluator):
    name = "latency"
    description = "Scores inversely proportional to response latency (lower is better)"

    async def evaluate(self, solution: Solution, context: dict[str, Any] | None = None) -> EvaluatorResult:
        ctx = context or {}
        max_ms = ctx.get("max_latency_ms", 30_000)
        raw = solution.latency_ms
        # Invert: lower latency → higher score
        score = max(0.0, 1.0 - (raw / max_ms))
        return EvaluatorResult(score=score, raw_score=raw, details={"max_latency_ms": max_ms})
