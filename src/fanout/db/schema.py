"""SQLite schema and initialization."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

TABLES_SQL = """\
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    prompt TEXT NOT NULL,
    config TEXT NOT NULL DEFAULT '{}',
    current_round INTEGER NOT NULL DEFAULT 0,
    total_rounds INTEGER NOT NULL DEFAULT 1,
    parent_run_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (parent_run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS solutions (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    round_num INTEGER NOT NULL DEFAULT 0,
    model TEXT NOT NULL,
    output TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms REAL NOT NULL DEFAULT 0.0,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    parent_solution_id TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id),
    FOREIGN KEY (parent_solution_id) REFERENCES solutions(id)
);

CREATE TABLE IF NOT EXISTS evaluations (
    id TEXT PRIMARY KEY,
    solution_id TEXT NOT NULL,
    evaluator TEXT NOT NULL,
    score REAL NOT NULL,
    raw_score REAL NOT NULL DEFAULT 0.0,
    details TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (solution_id) REFERENCES solutions(id)
);

CREATE INDEX IF NOT EXISTS idx_solutions_run_id ON solutions(run_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_solution_id ON evaluations(solution_id);
"""


def get_db_path(project_dir: Path | None = None) -> Path:
    """Return path to the SQLite database, creating .fanout/ if needed."""
    base = project_dir or Path.cwd()
    db_dir = base / ".fanout"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "fanout.db"


def init_db(db_path: Path | None = None) -> sqlite3.Connection:
    """Initialize the database and return a connection."""
    path = db_path or get_db_path()
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(TABLES_SQL)

    # Track schema version
    cur = conn.execute("SELECT version FROM schema_version")
    row = cur.fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
    conn.commit()
    return conn
