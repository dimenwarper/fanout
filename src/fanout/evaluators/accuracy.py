"""Accuracy evaluator — scores by exact or fuzzy reference matching."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from fanout.db.models import Solution
from fanout.evaluators.base import BaseEvaluator, EvaluatorResult, register_evaluator


@register_evaluator
class AccuracyEvaluator(BaseEvaluator):
    name = "accuracy"
    description = "Scores by similarity to a reference answer (exact or fuzzy match)"

    async def evaluate(self, solution: Solution, context: dict[str, Any] | None = None) -> EvaluatorResult:
        ctx = context or {}
        reference = ctx.get("reference", "")
        if not reference:
            return EvaluatorResult(score=0.0, raw_score=0.0, details={"error": "no reference provided"})

        output = solution.output.strip()
        ref = reference.strip()

        # Exact match
        if output == ref:
            return EvaluatorResult(score=1.0, raw_score=1.0, details={"match": "exact"})

        # Fuzzy match
        ratio = SequenceMatcher(None, output, ref).ratio()
        return EvaluatorResult(score=ratio, raw_score=ratio, details={"match": "fuzzy"})
