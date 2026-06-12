"""Fingerprint corpus stores — the set of known models a candidate is deduped against.

Two backends:
- NullFingerprintStore   — empty corpus (dedup always passes); the default.
- JsonlFingerprintStore  — local newline-delimited JSON; used by the CLI test harness.

A stored entry is a dict: {"key", "hotkey", "repo", "digest", "fingerprint"}.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Protocol

log = logging.getLogger(__name__)


class FingerprintStore(Protocol):
    def candidates(self) -> Iterable[dict]:
        """Yield stored fingerprint entries to compare a candidate against."""
        ...

    def add(self, key: str, fingerprint: dict, *, hotkey: str, repo: str, digest: str) -> None:
        """Record a fingerprint for ``key`` in the corpus."""
        ...


class NullFingerprintStore:
    """Empty corpus — no known models, so dedup never flags. Add is a no-op."""

    def candidates(self) -> Iterable[dict]:
        return ()

    def add(self, key: str, fingerprint: dict, *, hotkey: str, repo: str, digest: str) -> None:
        return None


class JsonlFingerprintStore:
    """Local JSONL-backed corpus for testing. Loads on init, appends on add."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._entries: list[dict] = []
        if self._path.exists():
            for line in self._path.read_text().splitlines():
                line = line.strip()
                if line:
                    self._entries.append(json.loads(line))
            log.info("fingerprint store: loaded %d entries from %s", len(self._entries), self._path)

    def candidates(self) -> Iterable[dict]:
        return list(self._entries)

    def add(self, key: str, fingerprint: dict, *, hotkey: str, repo: str, digest: str) -> None:
        entry = {"key": key, "hotkey": hotkey, "repo": repo, "digest": digest,
                 "fingerprint": fingerprint}
        self._entries.append(entry)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a") as fh:
            fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
