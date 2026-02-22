"""RSA (Recursive Self-Aggregation) strategy.

In round 0, samples independently. In round 1+, each new solution is generated
from a prompt that includes K randomly subsampled parent solutions, asking the
model to synthesize an improvement.
"""

from __future__ import annotations

import random
from typing import Any

from fanout.db.models import SolutionWithScores
from fanout.strategies.base import BaseStrategy, build_annotated_prompt, register_strategy


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
        a score-annotated aggregation prompt.
        """
        if round_num == 0 or not selected:
            return original_prompt

        k_agg = kwargs.get("k_agg", 3)
        k = min(k_agg, len(selected))

        prompts: list[str] = []
        for _ in range(n_samples):
            parents = random.sample(selected, k)
            prompt = build_annotated_prompt(original_prompt, parents)
            prompts.append(prompt)

        return prompts
