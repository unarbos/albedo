from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import SETTINGS
from .kings import King, utc_now


def _empty_state() -> dict[str, Any]:
    now = utc_now()
    return {
        "created_at": now,
        "updated_at": now,
        "benchmarks": {},
    }


def load_state(path: Path | None = None) -> dict[str, Any]:
    path = path or SETTINGS.state_path
    if not path.exists():
        return _empty_state()
    return json.loads(path.read_text())


def save_state(state: dict[str, Any], path: Path | None = None) -> None:
    path = path or SETTINGS.state_path
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = utc_now()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


@contextmanager
def state_lock() -> Iterator[None]:
    import fcntl

    SETTINGS.state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = SETTINGS.state_dir / ".state.lock"
    with lock_path.open("w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def select_next_king(kings: list[King], *, retry_failed: bool = False) -> King | None:
    state = load_state()
    benchmarks = state.setdefault("benchmarks", {})
    for king in kings:
        row = benchmarks.get(king.key)
        if row is None:
            return king
        status = row.get("status")
        if retry_failed and status == "failed":
            return king
    return None


def mark_running(king: King, *, run_id: str) -> None:
    with state_lock():
        state = load_state()
        state.setdefault("benchmarks", {})[king.key] = {
            "status": "running",
            "king": king.to_dict(),
            "run_id": run_id,
            "started_at": utc_now(),
        }
        save_state(state)


def mark_complete(king: King, result: dict[str, Any]) -> None:
    with state_lock():
        state = load_state()
        row = state.setdefault("benchmarks", {}).setdefault(king.key, {})
        row.update(result)
        row["status"] = "complete"
        row["completed_at"] = utc_now()
        save_state(state)


def mark_failed(king: King, error: str, partial: dict[str, Any] | None = None) -> None:
    with state_lock():
        state = load_state()
        row = state.setdefault("benchmarks", {}).setdefault(king.key, {})
        if partial:
            row.update(partial)
        row.setdefault("king", king.to_dict())
        row["status"] = "failed"
        row["error"] = error
        row["completed_at"] = utc_now()
        save_state(state)

