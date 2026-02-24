"""In-memory Channel implementation backed by plain dicts."""

from __future__ import annotations

import time
from fanout.channel import Channel


class MemoryChannel(Channel):
    """Ephemeral in-memory channel. Data lives only for the process lifetime."""

    def __init__(self) -> None:
        # topic -> key -> (payload, timestamp)
        self._data: dict[str, dict[str, tuple[dict, float]]] = {}
        # topic -> key -> {field: value, ...}
        self._indexes: dict[str, dict[str, dict[str, str]]] = {}

    def put(self, topic: str, key: str, payload: dict, **indexes: str) -> None:
        self._data.setdefault(topic, {})[key] = (payload, time.time())
        self._indexes.setdefault(topic, {})[key] = dict(indexes)

    def get(self, topic: str, key: str) -> dict | None:
        entry = self._data.get(topic, {}).get(key)
        if entry is None:
            return None
        return entry[0]

    def list(self, topic: str, **filters: str) -> list[dict]:
        bucket = self._data.get(topic, {})
        if not filters:
            items = sorted(bucket.items(), key=lambda kv: kv[1][1])
            return [payload for _, (payload, _ts) in items]

        idx = self._indexes.get(topic, {})
        matched = []
        for key, (payload, ts) in bucket.items():
            key_idx = idx.get(key, {})
            if all(key_idx.get(f) == v for f, v in filters.items()):
                matched.append((ts, payload))
        matched.sort(key=lambda x: x[0])
        return [payload for _ts, payload in matched]

    def delete(self, topic: str, key: str) -> bool:
        bucket = self._data.get(topic, {})
        if key not in bucket:
            return False
        del bucket[key]
        self._indexes.get(topic, {}).pop(key, None)
        return True
