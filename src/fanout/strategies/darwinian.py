"""Darwinian selection strategy — sigmoid scoring with novelty bonus.

Inspired by Imbue's Darwinian Evolver (https://github.com/imbue-ai/darwinian_evolver)
and the Darwin Gödel Machines paper (https://arxiv.org/abs/2505.22954).

Parent weight formula:
    w = sigmoid(sharpness × (score − midpoint)) × (1 / (1 + novelty_weight × n_selected))

where:
  - sigmoid(x) = 1 / (1 + exp(−x))
  - midpoint: fixed float or adaptive percentile string like "p75" (default)
  - n_selected: how many times this solution has already been used as a parent
                (tracked across rounds in WorkflowContext.selection_counts)
  - sharpness: steepness of the sigmoid — higher = more winner-take-all
  - novelty_weight: penalty for solutions already used as parents

Key differences vs plain weighted/top-k:
  - Sigmoid squashes extreme scores, reducing the pull of one runaway best
  - Novelty bonus prevents the same good solution from dominating every round
  - Adaptive midpoint (p75 by default) auto-scales to the current score distribution
"""

from __future__ import annotations

import math
import random
from typing import Any

from fanout.db.models import SolutionWithScores
from fanout.strategies.base import BaseStrategy, build_annotated_prompt, register_strategy


def _compute_midpoint(scores: list[float], midpoint: str | float) -> float:
    """Resolve midpoint — either a literal float or a 'pNN' percentile string."""
    if isinstance(midpoint, (int, float)):
        return float(midpoint)
    if isinstance(midpoint, str) and midpoint.startswith("p"):
        try:
            pct = int(midpoint[1:])
        except ValueError:
            pct = 75
        if not scores:
            return 0.5
        sorted_scores = sorted(scores)
        idx = max(0, min(len(sorted_scores) - 1, int(len(sorted_scores) * pct / 100)))
        return sorted_scores[idx]
    return 0.5  # fallback


@register_strategy
class DarwinianStrategy(BaseStrategy):
    """Sigmoid + novelty weighted selection.

    Parameters (passed as kwargs from select_solutions / select_step):
      k               — number of parents to select (default 3)
      sharpness       — sigmoid steepness (default 10.0)
      midpoint        — 'pNN' percentile or float threshold (default 'p75')
      novelty_weight  — penalty per prior parent use (default 1.0)
      selection_counts — dict[solution_id → n_times_selected] from WorkflowContext
    """

    name = "darwinian"
    description = (
        "Sigmoid-scaled selection with novelty bonus "
        "(inspired by Imbue's Darwinian Evolver / Darwin Gödel Machines)"
    )

    def select(
        self,
        candidates: list[SolutionWithScores],
        *,
        k: int = 3,
        sharpness: float = 10.0,
        midpoint: str | float = "p75",
        novelty_weight: float = 1.0,
        selection_counts: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> list[SolutionWithScores]:
        if not candidates:
            return []

        k = min(k, len(candidates))
        sel_counts = selection_counts or {}
        scores = [c.aggregate_score for c in candidates]
        mid = _compute_midpoint(scores, midpoint)

        # Compute per-candidate weight
        weights: list[float] = []
        for c in candidates:
            x = -sharpness * (c.aggregate_score - mid)
            x = max(-500.0, min(500.0, x))
            sigmoid = 1.0 / (1.0 + math.exp(x))
            n_used = sel_counts.get(c.solution.id, 0)
            novelty = 1.0 / (1.0 + novelty_weight * n_used)
            weights.append(sigmoid * novelty)

        # Weighted sampling without replacement
        selected: list[SolutionWithScores] = []
        pool = list(zip(candidates, weights))

        for _ in range(k):
            if not pool:
                break
            cands, ws = zip(*pool)
            # Guard against all-zero weights (e.g. all candidates used many times)
            total = sum(ws)
            if total <= 0:
                # Fall back to uniform sampling
                chosen = random.choice(list(cands))
            else:
                chosen = random.choices(list(cands), weights=list(ws), k=1)[0]
            selected.append(chosen)
            pool = [(c, w) for c, w in pool if c is not chosen]

        return selected

    def build_prompts(
        self,
        original_prompt: str,
        selected: list[SolutionWithScores],
        round_num: int,
        n_samples: int,
        **kwargs: Any,
    ) -> str | list[str]:
        """Build next-round prompt, explicitly framing failures as cases to fix.

        Inherits error-aware annotation from build_annotated_prompt, but adds
        Darwinian framing: each failure is a 'case that needs to be resolved',
        mirroring the mutator's role in the Darwinian Evolver.
        """
        if round_num == 0 or not selected:
            return original_prompt

        return build_annotated_prompt(
            original_prompt,
            selected,
            instruction=(
                "Produce an improved solution. "
                "Treat each ERRORS block as a specific failure case to diagnose and fix. "
                "Try a fundamentally different approach if the current ones share the same weaknesses."
            ),
        )
