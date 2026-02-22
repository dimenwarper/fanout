"""Model sets — named ensembles of models for weighted-random sampling."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


class ModelEntry(BaseModel):
    """A model with a sampling weight."""

    model: str
    weight: float = 1.0


class ModelSet(BaseModel):
    """A named collection of weighted models."""

    name: str
    models: list[ModelEntry]


# ── Built-in presets ─────────────────────────────────────

BUILTIN_SETS: dict[str, ModelSet] = {
    "diverse": ModelSet(
        name="diverse",
        models=[
            ModelEntry(model="openai/gpt-4o-mini"),
            ModelEntry(model="anthropic/claude-haiku-4"),
            ModelEntry(model="meta-llama/llama-3.1-8b-instruct"),
            ModelEntry(model="openai/gpt-4o"),
            ModelEntry(model="anthropic/claude-sonnet-4"),
        ],
    ),
    "small": ModelSet(
        name="small",
        models=[
            ModelEntry(model="openai/gpt-4o-mini"),
            ModelEntry(model="anthropic/claude-haiku-4"),
            ModelEntry(model="meta-llama/llama-3.1-8b-instruct"),
            ModelEntry(model="google/gemini-2.0-flash"),
        ],
    ),
    "large": ModelSet(
        name="large",
        models=[
            ModelEntry(model="openai/gpt-4o"),
            ModelEntry(model="anthropic/claude-sonnet-4"),
            ModelEntry(model="google/gemini-2.5-pro"),
            ModelEntry(model="meta-llama/llama-3.1-70b-instruct"),
        ],
    ),
    "coding": ModelSet(
        name="coding",
        models=[
            ModelEntry(model="anthropic/claude-opus-4.6", weight=2.0),
            ModelEntry(model="openai/gpt-5.2-codex", weight=2.0),
            ModelEntry(model="deepseek/deepseek-v3.2", weight=1.5),
            ModelEntry(model="qwen/qwen3-coder", weight=1.0),
        ],
    ),
    "math-proving": ModelSet(
        name="math-proving",
        models=[
            ModelEntry(model="deepseek/deepseek-r1", weight=2.0),
            ModelEntry(model="openai/o3", weight=2.0),
            ModelEntry(model="anthropic/claude-opus-4.6", weight=1.5),
            ModelEntry(model="qwen/qwen3-coder", weight=1.0),
        ],
    ),
}


# ── Loading & lookup ─────────────────────────────────────

def load_model_sets(config_dir: Path | None = None) -> dict[str, ModelSet]:
    """Merge built-in presets with user-defined sets from .fanout/model_sets.toml."""
    sets = dict(BUILTIN_SETS)

    toml_path = (config_dir or Path.cwd()) / ".fanout" / "model_sets.toml"
    if toml_path.is_file():
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        for name, cfg in data.get("sets", {}).items():
            entries = [ModelEntry(**m) for m in cfg.get("models", [])]
            if entries:
                sets[name] = ModelSet(name=name, models=entries)

    return sets


def get_model_set(name: str, config_dir: Path | None = None) -> ModelSet:
    """Look up a model set by name (builtins + user config)."""
    sets = load_model_sets(config_dir)
    if name not in sets:
        available = ", ".join(sorted(sets))
        raise KeyError(f"Unknown model set {name!r}. Available: {available}")
    return sets[name]


def pick_model(model_set: ModelSet) -> str:
    """Weighted random choice of one model from the set."""
    models = [e.model for e in model_set.models]
    weights = [e.weight for e in model_set.models]
    return random.choices(models, weights=weights, k=1)[0]


def pick_models(model_set: ModelSet, n: int) -> list[str]:
    """Draw N models with replacement (weighted random)."""
    models = [e.model for e in model_set.models]
    weights = [e.weight for e in model_set.models]
    return random.choices(models, weights=weights, k=n)
