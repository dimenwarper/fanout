"""AlphaEvolve strategy.

Combines score-aware tournament selection with diversity preservation,
score-annotated aggregation prompts, and score-biased parent subsampling.
"""

from __future__ import annotations

import random
from typing import Any

from fanout.db.models import SolutionWithScores
from fanout.strategies.base import BaseStrategy, register_strategy


def _pick_diverse(
    candidates: list[SolutionWithScores], n: int
) -> list[SolutionWithScores]:
    """Pick up to *n* diverse candidates, preferring the best per unseen model."""
    if n <= 0:
        return []
    # Best remaining candidate per model
    best_per_model: dict[str, SolutionWithScores] = {}
    for c in candidates:
        model = c.solution.model
        if model not in best_per_model or c.aggregate_score > best_per_model[model].aggregate_score:
            best_per_model[model] = c
    # Take best-per-model first (sorted by score), then fill from remainder
    diverse = sorted(best_per_model.values(), key=lambda c: c.aggregate_score, reverse=True)[:n]
    if len(diverse) < n:
        seen = set(id(c) for c in diverse)
        for c in sorted(candidates, key=lambda c: c.aggregate_score, reverse=True):
            if id(c) not in seen:
                diverse.append(c)
                seen.add(id(c))
                if len(diverse) >= n:
                    break
    return diverse


@register_strategy
class AlphaEvolveStrategy(BaseStrategy):
    name = "alphaevolve"
    description = (
        "AlphaEvolve — score-aware selection + annotated aggregation prompts + biased subsampling"
    )

    def select(
        self, candidates: list[SolutionWithScores], **kwargs: Any
    ) -> list[SolutionWithScores]:
        """Tournament selection with diversity preservation.

        Keep top elites, then fill remaining slots with diverse solutions
        (best per model not already selected).
        """
        k = kwargs.get("k", 3)
        if len(candidates) <= k:
            return candidates

        candidates_sorted = sorted(
            candidates, key=lambda c: c.aggregate_score, reverse=True
        )
        n_elites = max(1, k // 2)
        elites = candidates_sorted[:n_elites]

        remaining = [c for c in candidates_sorted if c not in elites]
        diverse = _pick_diverse(remaining, k - len(elites))

        return elites + diverse

    def build_prompts(
        self,
        original_prompt: str,
        selected: list[SolutionWithScores],
        round_num: int,
        n_samples: int,
        **kwargs: Any,
    ) -> str | list[str]:
        """Score-annotated aggregation prompts with biased subsampling.

        Round 0: return original prompt (independent sampling).
        Round 1+: for each of n_samples, pick K parents via score-biased
        sampling and build a score-annotated aggregation prompt.
        """
        if round_num == 0 or not selected:
            return original_prompt

        k_agg = kwargs.get("k_agg", 3)
        k = min(k_agg, len(selected))

        prompts: list[str] = []
        for _ in range(n_samples):
            parents = _biased_sample(selected, k)
            prompt = _build_annotated_prompt(original_prompt, parents)
            prompts.append(prompt)

        return prompts


def _biased_sample(
    selected: list[SolutionWithScores], k: int
) -> list[SolutionWithScores]:
    """Score-biased sampling: higher-scoring parents more likely to be picked."""
    weights = [s.aggregate_score + 0.05 for s in selected]
    sampled = random.choices(selected, weights=weights, k=k)
    # Deduplicate while preserving order
    seen: set[int] = set()
    unique: list[SolutionWithScores] = []
    for s in sampled:
        sid = id(s)
        if sid not in seen:
            seen.add(sid)
            unique.append(s)
    return unique


def _build_annotated_prompt(
    original_prompt: str,
    parents: list[SolutionWithScores],
) -> str:
    """Build a score-annotated aggregation prompt."""
    parents_sorted = sorted(parents, key=lambda p: p.aggregate_score, reverse=True)
    best_score = parents_sorted[0].aggregate_score if parents_sorted else 0.0

    parts = [
        f"Original task: {original_prompt}",
        "",
        f"You are shown {len(parents_sorted)} previous attempts, ranked by score (higher = better, max 1.0).",
        "Study what makes the top-scoring solutions work and what makes lower-scoring ones fail.",
        "Produce an improved solution that builds on the strengths and fixes the weaknesses.",
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

    parts.append("Produce ONLY the improved solution code \u2014 no explanations:")
    return "\n".join(parts)
