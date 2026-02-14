"""Base evaluator ABC and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from fanout.db.models import Evaluation, Solution

_REGISTRY: dict[str, type[BaseEvaluator]] = {}


class EvaluatorResult:
    """Result from an evaluator."""

    def __init__(self, score: float, raw_score: float = 0.0, details: dict[str, Any] | None = None):
        self.score = max(0.0, min(1.0, score))  # clamp to [0, 1]
        self.raw_score = raw_score
        self.details = details or {}


class BaseEvaluator(ABC):
    """Abstract base for all evaluators."""

    name: str = "base"
    description: str = ""

    @abstractmethod
    async def evaluate(self, solution: Solution, context: dict[str, Any] | None = None) -> EvaluatorResult:
        """Score a solution. Returns normalized score in [0, 1]."""
        ...

    def to_evaluation(self, solution: Solution, result: EvaluatorResult) -> Evaluation:
        return Evaluation(
            solution_id=solution.id,
            evaluator=self.name,
            score=result.score,
            raw_score=result.raw_score,
            details=result.details,
        )


def register_evaluator(cls: type[BaseEvaluator]) -> type[BaseEvaluator]:
    """Class decorator to register an evaluator."""
    _REGISTRY[cls.name] = cls
    return cls


def get_evaluator(name: str) -> BaseEvaluator:
    """Instantiate a registered evaluator by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown evaluator: {name!r}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]()


def list_evaluators() -> dict[str, type[BaseEvaluator]]:
    """Return all registered evaluators."""
    return dict(_REGISTRY)
