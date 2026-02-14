"""Cost evaluator — scores inversely proportional to USD cost."""

from __future__ import annotations

from typing import Any

from fanout.db.models import Solution
from fanout.evaluators.base import BaseEvaluator, EvaluatorResult, register_evaluator


@register_evaluator
class CostEvaluator(BaseEvaluator):
    name = "cost"
    description = "Scores inversely proportional to token cost (cheaper is better)"

    async def evaluate(self, solution: Solution, context: dict[str, Any] | None = None) -> EvaluatorResult:
        ctx = context or {}
        max_cost = ctx.get("max_cost_usd", 1.0)
        raw = solution.cost_usd
        score = max(0.0, 1.0 - (raw / max_cost))
        return EvaluatorResult(score=score, raw_score=raw, details={"max_cost_usd": max_cost})
