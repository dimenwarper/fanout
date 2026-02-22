"""RSA (Recursive Self-Aggregation) strategy.

In round 0, samples independently. In round 1+, each new solution is generated
from a prompt that includes K randomly subsampled parent solutions, asking the
model to synthesize an improvement.
"""

from __future__ import annotations

import random
from typing import Any

from fanout.db.models import SolutionWithScores
from fanout.strategies.base import BaseStrategy, register_strategy


@register_strategy
class RSAStrategy(BaseStrategy):
    name = "rsa"
    description = "Recursive Self-Aggregation — feed K parent solutions back into each prompt"

    def select(self, candidates: list[SolutionWithScores], **kwargs: Any) -> list[SolutionWithScores]:
        """Return all candidates — RSA doesn't filter by score, aggregation is the mechanism."""
        return candidates

    def build_prompts(
        self,
        original_prompt: str,
        selected: list[SolutionWithScores],
        round_num: int,
        n_samples: int,
        **kwargs: Any,
    ) -> str | list[str]:
        """Build per-solution aggregation prompts for the next round.

        Round 0: return original prompt (independent sampling).
        Round 1+: for each of n_samples, randomly pick K parents and build
        an aggregation prompt.
        """
        if round_num == 0 or not selected:
            return original_prompt

        k_agg = kwargs.get("k_agg", 3)
        k = min(k_agg, len(selected))

        prompts: list[str] = []
        for _ in range(n_samples):
            parents = random.sample(selected, k)
            prompt = _build_aggregation_prompt(original_prompt, parents)
            prompts.append(prompt)

        return prompts


def _build_aggregation_prompt(
    original_prompt: str,
    parents: list[SolutionWithScores],
) -> str:
    parents_sorted = sorted(parents, key=lambda p: p.aggregate_score, reverse=True)
    best_score = parents_sorted[0].aggregate_score if parents_sorted else 0.0

    parts = [
        f"Original task: {original_prompt}",
        "",
        f"You are shown {len(parents_sorted)} previous solutions, ranked by score (higher = better, max 1.0).",
        "Analyze them and produce an improved solution that combines their best ideas and addresses weaknesses.",
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
