"""Base strategy ABC and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from fanout.db.models import SolutionWithScores

_REGISTRY: dict[str, type[BaseStrategy]] = {}


def build_annotated_prompt(
    original_prompt: str,
    parents: list[SolutionWithScores],
    instruction: str = "Produce an improved solution that combines their best ideas and addresses weaknesses.",
) -> str:
    """Build a score-annotated aggregation prompt from parent solutions.

    Shared helper used by all strategies that feed parents into the next round.
    Parents are sorted by score (best first) and annotated with score + model.
    """
    parents_sorted = sorted(parents, key=lambda p: p.aggregate_score, reverse=True)
    best_score = parents_sorted[0].aggregate_score if parents_sorted else 0.0

    parts = [
        f"Original task: {original_prompt}",
        "",
        f"You are shown {len(parents_sorted)} previous solutions, ranked by score (higher = better, max 1.0).",
        instruction,
        "",
    ]
    for i, parent in enumerate(parents_sorted, 1):
        score = parent.aggregate_score
        model = parent.solution.model
        label = f"Solution {i} (score: {score:.2f}, model: {model})"
        if score == best_score:
            label += " \u2605 BEST"
        parts.append(f"=== {label} ===")
        parts.append(parent.solution.output)
        parts.append("")

    parts.append("Provide your improved solution:")
    return "\n".join(parts)


class BaseStrategy(ABC):
    """Abstract base for selection strategies."""

    name: str = "base"
    description: str = ""

    @abstractmethod
    def select(self, candidates: list[SolutionWithScores], **kwargs: Any) -> list[SolutionWithScores]:
        """Select from scored candidates for the next round."""
        ...

    def build_prompts(
        self,
        original_prompt: str,
        selected: list[SolutionWithScores],
        round_num: int,
        n_samples: int,
        **kwargs: Any,
    ) -> str | list[str]:
        """Build prompt(s) for the next round.

        Default: broadcast a single score-annotated prompt showing all
        selected parents to every sample. Round 0 returns the original prompt.
        """
        if round_num == 0 or not selected:
            return original_prompt
        return build_annotated_prompt(original_prompt, selected)


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
