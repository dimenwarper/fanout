"""SQLite-backed Channel implementation."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from fanout.channel import Channel
from fanout.db.schema import get_db_path

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS messages (
    topic TEXT NOT NULL,
    key TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (topic, key)
);

CREATE TABLE IF NOT EXISTS indexes (
    topic TEXT NOT NULL,
    key TEXT NOT NULL,
    field TEXT NOT NULL,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_indexes_lookup ON indexes(topic, field, value);
CREATE INDEX IF NOT EXISTS idx_indexes_key ON indexes(topic, key);
"""


class SqliteChannel(Channel):
    """Channel backed by a single SQLite database with generic messages + indexes tables."""

    def __init__(self, db_path: Path | None = None) -> None:
        path = db_path or get_db_path()
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(_SCHEMA_SQL)
        self.conn.commit()

    def put(self, topic: str, key: str, payload: dict, **indexes: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT OR REPLACE INTO messages (topic, key, payload, created_at) VALUES (?, ?, ?, ?)",
            (topic, key, json.dumps(payload, default=str), now),
        )
        # Replace index rows
        self.conn.execute("DELETE FROM indexes WHERE topic = ? AND key = ?", (topic, key))
        if indexes:
            self.conn.executemany(
                "INSERT INTO indexes (topic, key, field, value) VALUES (?, ?, ?, ?)",
                [(topic, key, field, value) for field, value in indexes.items()],
            )
        self.conn.commit()

    def get(self, topic: str, key: str) -> dict | None:
        row = self.conn.execute(
            "SELECT payload FROM messages WHERE topic = ? AND key = ?", (topic, key)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def list(self, topic: str, **filters: str) -> list[dict]:
        if not filters:
            rows = self.conn.execute(
                "SELECT payload FROM messages WHERE topic = ? ORDER BY created_at",
                (topic,),
            ).fetchall()
        else:
            # Intersect index matches for each filter field
            clauses = []
            params: list[str] = []
            for field, value in filters.items():
                clauses.append(
                    "SELECT key FROM indexes WHERE topic = ? AND field = ? AND value = ?"
                )
                params.extend([topic, field, value])

            keys_sql = " INTERSECT ".join(clauses)
            rows = self.conn.execute(
                f"SELECT payload FROM messages WHERE topic = ? AND key IN ({keys_sql}) ORDER BY created_at",
                [topic, *params],
            ).fetchall()

        return [json.loads(r[0]) for r in rows]

    def delete(self, topic: str, key: str) -> bool:
        cur = self.conn.execute(
            "DELETE FROM messages WHERE topic = ? AND key = ?", (topic, key)
        )
        self.conn.execute("DELETE FROM indexes WHERE topic = ? AND key = ?", (topic, key))
        self.conn.commit()
        return cur.rowcount > 0
