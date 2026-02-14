"""Base strategy ABC and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from fanout.db.models import SolutionWithScores

_REGISTRY: dict[str, type[BaseStrategy]] = {}


class BaseStrategy(ABC):
    """Abstract base for selection strategies."""

    name: str = "base"
    description: str = ""

    @abstractmethod
    def select(self, candidates: list[SolutionWithScores], **kwargs: Any) -> list[SolutionWithScores]:
        """Select from scored candidates for the next round."""
        ...


def register_strategy(cls: type[BaseStrategy]) -> type[BaseStrategy]:
    """Class decorator to register a strategy."""
    _REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str) -> BaseStrategy:
    """Instantiate a registered strategy by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown strategy: {name!r}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]()


def list_strategies() -> dict[str, type[BaseStrategy]]:
    """Return all registered strategies."""
    return dict(_REGISTRY)
