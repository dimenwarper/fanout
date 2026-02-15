"""Base materializer ABC and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from fanout.db.models import Solution

_REGISTRY: dict[str, type[BaseMaterializer]] = {}


class BaseMaterializer(ABC):
    """Abstract base for all materializers."""

    name: str = "base"
    description: str = ""

    @abstractmethod
    async def materialize(self, solution: Solution, workspace: Path, context: dict) -> Path:
        """Write/apply solution output to workspace, return path for eval script."""
        ...

    async def cleanup(self, workspace: Path) -> None:
        """Optional cleanup after evaluation. Default: no-op."""

def register_materializer(cls: type[BaseMaterializer]) -> type[BaseMaterializer]:
    """Class decorator to register a materializer."""
    _REGISTRY[cls.name] = cls
    return cls


def get_materializer(name: str) -> BaseMaterializer:
    """Instantiate a registered materializer by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown materializer: {name!r}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]()


def list_materializers() -> dict[str, type[BaseMaterializer]]:
    """Return all registered materializers."""
    return dict(_REGISTRY)
