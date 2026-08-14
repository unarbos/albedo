from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS processed_runs(
  run_id TEXT PRIMARY KEY,
  rows INTEGER NOT NULL,
  source TEXT NOT NULL,
  ingested_at TEXT NOT NULL,
  started_at TEXT
);
CREATE TABLE IF NOT EXISTS chunks(
  path TEXT PRIMARY KEY,
  model TEXT NOT NULL,
  created_at TEXT NOT NULL,
  uploaded_at TEXT
);
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY,
  value TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class State:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def is_processed(self, run_id: str) -> bool:
        cur = self.conn.execute("SELECT 1 FROM processed_runs WHERE run_id=?", (run_id,))
        return cur.fetchone() is not None

    def mark_processed(
        self,
        run_id: str,
        rows: int,
        source: str,
        ingested_at: str | None = None,
        started_at: str | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO processed_runs(run_id, rows, source, ingested_at, started_at)"
            " VALUES (?,?,?,?,?)",
            (run_id, rows, source, ingested_at or _now(), started_at),
        )
        self.conn.commit()

    def processed_count(self) -> int:
        return self.conn.execute("SELECT count(*) FROM processed_runs").fetchone()[0]

    def chunk_exists(self, path: str) -> bool:
        cur = self.conn.execute("SELECT 1 FROM chunks WHERE path=?", (path,))
        return cur.fetchone() is not None

    def record_chunk(self, path: str, model: str, uploaded_at: str | None = None) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO chunks(path, model, created_at, uploaded_at) VALUES (?,?,?,?)",
            (path, model, _now(), uploaded_at),
        )
        self.conn.commit()

    def mark_uploaded(self, path: str) -> None:
        self.conn.execute("UPDATE chunks SET uploaded_at=? WHERE path=?", (_now(), path))
        self.conn.commit()

    def pending_uploads(self) -> list[tuple[str, str]]:
        cur = self.conn.execute(
            "SELECT path, model FROM chunks WHERE uploaded_at IS NULL ORDER BY path"
        )
        return cur.fetchall()

    def chunk_models(self) -> list[str]:
        cur = self.conn.execute("SELECT DISTINCT model FROM chunks ORDER BY model")
        return [r[0] for r in cur.fetchall()]

    def chunk_count(self, model: str) -> int:
        return self.conn.execute("SELECT count(*) FROM chunks WHERE model=?", (model,)).fetchone()[
            0
        ]

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?,?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.conn.commit()
