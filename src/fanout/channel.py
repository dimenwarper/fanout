"""Abstract Channel interface for inter-agent communication."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Channel(ABC):
    """Minimal put/get/list/delete interface over named topics."""

    @abstractmethod
    def put(self, topic: str, key: str, payload: dict, **indexes: str) -> None:
        """Store *payload* under *topic*/*key*, with optional queryable indexes."""

    @abstractmethod
    def get(self, topic: str, key: str) -> dict | None:
        """Return the payload for *topic*/*key*, or ``None`` if missing."""

    @abstractmethod
    def list(self, topic: str, **filters: str) -> list[dict]:
        """Return all payloads in *topic*, optionally filtered by index equality."""

    @abstractmethod
    def delete(self, topic: str, key: str) -> bool:
        """Delete *topic*/*key*. Return ``True`` if it existed."""
