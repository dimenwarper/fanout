"""Redis-backed Channel implementation."""

from __future__ import annotations

import json
import time

import redis

from fanout.channel import Channel


class RedisChannel(Channel):
    """Channel backed by Redis with key-prefix scheme.

    Key layout:
        {prefix}msg:{topic}:{key}          — JSON payload
        {prefix}topic:{topic}              — sorted set (score = timestamp)
        {prefix}idx:{topic}:{field}:{value} — set of keys
        {prefix}ikeys:{topic}:{key}        — set of "field:value" pairs for cleanup
    """

    def __init__(self, url: str = "redis://localhost:6379", prefix: str = "fanout:") -> None:
        self.r = redis.Redis.from_url(url, decode_responses=True)
        self.prefix = prefix

    # ── helpers ────────────────────────────────────────────

    def _msg_key(self, topic: str, key: str) -> str:
        return f"{self.prefix}msg:{topic}:{key}"

    def _topic_key(self, topic: str) -> str:
        return f"{self.prefix}topic:{topic}"

    def _idx_key(self, topic: str, field: str, value: str) -> str:
        return f"{self.prefix}idx:{topic}:{field}:{value}"

    def _ikeys_key(self, topic: str, key: str) -> str:
        return f"{self.prefix}ikeys:{topic}:{key}"

    # ── Channel interface ─────────────────────────────────

    def put(self, topic: str, key: str, payload: dict, **indexes: str) -> None:
        pipe = self.r.pipeline()

        # Clean up old index entries if this key already exists
        old_ikeys = self.r.smembers(self._ikeys_key(topic, key))
        for entry in old_ikeys:
            field, value = entry.split(":", 1)
            pipe.srem(self._idx_key(topic, field, value), key)
        if old_ikeys:
            pipe.delete(self._ikeys_key(topic, key))

        # Store payload
        pipe.set(self._msg_key(topic, key), json.dumps(payload, default=str))

        # Add to topic sorted set with timestamp score
        pipe.zadd(self._topic_key(topic), {key: time.time()})

        # Add new index entries
        if indexes:
            ikeys_key = self._ikeys_key(topic, key)
            for field, value in indexes.items():
                pipe.sadd(self._idx_key(topic, field, value), key)
                pipe.sadd(ikeys_key, f"{field}:{value}")

        pipe.execute()

    def get(self, topic: str, key: str) -> dict | None:
        raw = self.r.get(self._msg_key(topic, key))
        if raw is None:
            return None
        return json.loads(raw)

    def list(self, topic: str, **filters: str) -> list[dict]:
        if not filters:
            keys = self.r.zrange(self._topic_key(topic), 0, -1)
        else:
            idx_keys = [
                self._idx_key(topic, field, value)
                for field, value in filters.items()
            ]
            if len(idx_keys) == 1:
                keys = list(self.r.smembers(idx_keys[0]))
            else:
                keys = list(self.r.sinter(idx_keys))

        if not keys:
            return []

        msg_keys = [self._msg_key(topic, k) for k in keys]
        values = self.r.mget(msg_keys)
        return [json.loads(v) for v in values if v is not None]

    def delete(self, topic: str, key: str) -> bool:
        if not self.r.exists(self._msg_key(topic, key)):
            return False

        pipe = self.r.pipeline()

        # Clean up index entries
        old_ikeys = self.r.smembers(self._ikeys_key(topic, key))
        for entry in old_ikeys:
            field, value = entry.split(":", 1)
            pipe.srem(self._idx_key(topic, field, value), key)

        pipe.delete(self._msg_key(topic, key))
        pipe.delete(self._ikeys_key(topic, key))
        pipe.zrem(self._topic_key(topic), key)
        pipe.execute()
        return True
